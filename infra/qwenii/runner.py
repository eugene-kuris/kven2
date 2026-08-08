#!/usr/bin/env python3
import hashlib
import json
import logging
import os
import re
import resource
import signal
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path('/etc/kven-qwenii-runner.json')
AUTH_RE = re.compile(r'^KVEN-QWENII-AUTH-([A-Za-z0-9._-]{8,96})\.json$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
TERMINAL = {'SUCCEEDED', 'FAILED', 'TIMED_OUT'}
FINAL = TERMINAL | {'RETURNED', 'REJECTED'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('kven-qwenii-runner')


def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def secure_regular_root_file(path):
    st = path.lstat()
    return stat.S_ISREG(st.st_mode) and st.st_uid == 0 and (stat.S_IMODE(st.st_mode) & 0o022) == 0


def receipt_for(path):
    return Path(str(path) + '.receipt.json')


def verify_receipt(target, receipt_path, corr, cfg, expected_sha=None):
    if not target.is_file() or not receipt_path.is_file():
        raise FileNotFoundError('artifact_or_receipt_missing')
    if not secure_regular_root_file(target) or not secure_regular_root_file(receipt_path):
        raise ValueError('artifact_or_receipt_permissions')
    rec = load_json(receipt_path)
    required = {
        'correlation_id', 'sender', 'subject', 'message_id', 'filename',
        'path', 'size', 'sha256', 'saved_mode', 'automatic_execution'
    }
    if set(rec) != required:
        raise ValueError('receipt_schema')
    if rec['correlation_id'] != corr:
        raise ValueError('receipt_correlation')
    if rec['sender'].lower() != cfg['accepted_sender'].lower():
        raise ValueError('receipt_sender')
    if rec['subject'] != f'[KVEN-BRIDGE] FILE {corr}':
        raise ValueError('receipt_subject')
    if rec['filename'] != target.name or rec['path'] != str(target):
        raise ValueError('receipt_path')
    if rec['saved_mode'] != '0644' or rec['automatic_execution'] is not False:
        raise ValueError('receipt_transport_semantics')
    actual_size = target.stat().st_size
    actual_sha = sha256_file(target)
    if rec['size'] != actual_size or rec['sha256'] != actual_sha:
        raise ValueError('receipt_integrity')
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError('envelope_payload_hash')
    return rec, actual_sha


def state_path(cfg, corr):
    return Path(cfg['state_dir']) / 'tasks' / f'{corr}.json'


def read_state(cfg, corr):
    p = state_path(cfg, corr)
    return load_json(p) if p.exists() else None


def write_state(cfg, rec):
    rec['updated_at'] = now_utc()
    atomic_json(state_path(cfg, rec['correlation_id']), rec)


def reject(cfg, rec, reason):
    rec['state'] = 'REJECTED'
    rec['validation_error'] = reason
    write_state(cfg, rec)
    log.warning('rejected corr=%s reason=%s', rec['correlation_id'], reason)


def validate_envelope(cfg, auth_path, corr, rec):
    auth_receipt, auth_sha = verify_receipt(auth_path, receipt_for(auth_path), corr, cfg)
    if rec.get('auth_sha256') and rec['auth_sha256'] != auth_sha:
        raise ValueError('authorization_artifact_changed')
    env = load_json(auth_path)
    if set(env) != {'protocol', 'authorization', 'task_id', 'correlation_id', 'payload', 'timeout_seconds'}:
        raise ValueError('envelope_schema')
    if env['protocol'] != 'KVEN-QWENII-TASK/1':
        raise ValueError('protocol')
    if env['authorization'] != 'EXECUTE_AS_QWENII':
        raise ValueError('authorization')
    if env['task_id'] != corr or env['correlation_id'] != corr:
        raise ValueError('envelope_correlation')
    if not isinstance(env['payload'], dict) or set(env['payload']) != {'filename', 'sha256'}:
        raise ValueError('payload_schema')
    payload_name = env['payload']['filename']
    payload_sha = env['payload']['sha256']
    expected_name = f'KVEN-QWENII-PAYLOAD-{corr}.sh'
    if payload_name != expected_name or not SHA_RE.fullmatch(str(payload_sha)):
        raise ValueError('payload_identity')
    timeout = env['timeout_seconds']
    if type(timeout) is not int or not (1 <= timeout <= int(cfg['max_timeout_seconds'])):
        raise ValueError('timeout')
    payload = Path(cfg['inbox_dir']) / payload_name
    payload_receipt = receipt_for(payload)
    if not payload.exists() or not payload_receipt.exists():
        return None
    _, actual_payload_sha = verify_receipt(payload, payload_receipt, corr, cfg, payload_sha)
    rec.update({
        'task_id': corr,
        'auth_filename': auth_path.name,
        'auth_sha256': auth_sha,
        'auth_message_id': auth_receipt['message_id'],
        'payload_filename': payload.name,
        'payload_sha256': actual_payload_sha,
        'payload_message_id': load_json(payload_receipt)['message_id'],
        'timeout_seconds': timeout,
        'state': 'VALIDATED',
        'validated_at': now_utc(),
    })
    write_state(cfg, rec)
    log.info('validated corr=%s payload=%s sha256=%s', corr, payload.name, actual_payload_sha)
    return payload


def bounded_read(path, limit):
    data = path.read_bytes() if path.exists() else b''
    clipped = len(data) > limit
    data = data[:limit]
    text = data.decode('utf-8', errors='replace')
    return text, clipped


def render_result(cfg, rec):
    result_dir = Path(cfg['results_dir'])
    stdout_path = result_dir / f"{rec['correlation_id']}.stdout.txt"
    stderr_path = result_dir / f"{rec['correlation_id']}.stderr.txt"
    result_path = result_dir / f"{rec['correlation_id']}.result.txt"
    out_text, out_clip = bounded_read(stdout_path, int(cfg['inline_output_bytes']))
    err_text, err_clip = bounded_read(stderr_path, int(cfg['inline_output_bytes']))
    lines = [
        'TASK: WORK-KVEN-MAIL-001C',
        f"TASK_ID: {rec['task_id']}",
        f"CORRELATION_ID: {rec['correlation_id']}",
        f"INPUT_SHA256: {rec['payload_sha256']}",
        f"EXECUTION_IDENTITY: {cfg['execution_user']}",
        f"EXECUTION_COUNT: {rec.get('execution_count', 0)}",
        f"TERMINAL_STATE: {rec.get('terminal_state', rec.get('state'))}",
        f"START_TIME: {rec.get('started_at', '')}",
        f"FINISH_TIME: {rec.get('finished_at', '')}",
        f"EXIT_CODE: {rec.get('exit_code', '')}",
        f"STDOUT_PATH: {stdout_path}",
        f"STDOUT_SHA256: {sha256_file(stdout_path) if stdout_path.exists() else ''}",
        f"STDERR_PATH: {stderr_path}",
        f"STDERR_SHA256: {sha256_file(stderr_path) if stderr_path.exists() else ''}",
        f"RESULT_PATH: {result_path}",
        'RETURN_STATUS_AT_RENDER: PENDING',
    ]
    if rec.get('failure_reason'):
        lines.append(f"FAILURE_REASON: {rec['failure_reason']}")
    lines += [
        '', '===== STDOUT =====', out_text,
        '[STDOUT TRUNCATED]' if out_clip else '[STDOUT COMPLETE]',
        '', '===== STDERR =====', err_text,
        '[STDERR TRUNCATED]' if err_clip else '[STDERR COMPLETE]', ''
    ]
    atomic_text(result_path, '\n'.join(lines), 0o644)
    rec['result_path'] = str(result_path)
    rec['result_sha256'] = sha256_file(result_path)
    write_state(cfg, rec)
    return result_path


def limit_child_files(cfg):
    cap = int(cfg['child_file_size_bytes'])
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))


def run_task(cfg, rec, payload):
    corr = rec['correlation_id']
    result_dir = Path(cfg['results_dir'])
    result_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = result_dir / f'{corr}.stdout.txt'
    stderr_path = result_dir / f'{corr}.stderr.txt'
    for p in (stdout_path, stderr_path):
        if p.exists():
            raise RuntimeError('output_path_already_exists')
    rec['execution_count'] = int(rec.get('execution_count', 0)) + 1
    rec['state'] = 'RUNNING'
    rec['started_at'] = now_utc()
    write_state(cfg, rec)
    log.info('running corr=%s as=%s', corr, cfg['execution_user'])
    timed_out = False
    rc = None
    with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
        os.chmod(stdout_path, 0o644)
        os.chmod(stderr_path, 0o644)
        args = [
            cfg['runuser_path'], '-u', cfg['execution_user'], '--',
            '/usr/bin/env', '-i',
            'HOME=/home/qwenii', 'USER=qwenii', 'LOGNAME=qwenii',
            'PATH=/usr/bin:/bin', f'KVEN_TASK_ID={corr}', f'KVEN_CORRELATION_ID={corr}',
            '/bin/bash', str(payload),
        ]
        try:
            proc = subprocess.Popen(
                args, cwd=cfg['work_dir'], stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, start_new_session=True,
                preexec_fn=lambda: limit_child_files(cfg),
            )
            rec['launcher_pid'] = proc.pid
            write_state(cfg, rec)
            try:
                rc = proc.wait(timeout=int(rec['timeout_seconds']))
                # A bounded task must not leave descendants behind after its main shell exits.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    rc = proc.wait()
        except Exception as exc:
            rec['failure_reason'] = f'launch_error:{type(exc).__name__}'
            rc = None
    rec['finished_at'] = now_utc()
    rec['exit_code'] = rc
    if timed_out:
        rec['state'] = rec['terminal_state'] = 'TIMED_OUT'
    elif rc == 0:
        rec['state'] = rec['terminal_state'] = 'SUCCEEDED'
    else:
        rec['state'] = rec['terminal_state'] = 'FAILED'
    write_state(cfg, rec)
    result = render_result(cfg, rec)
    log.info('terminal corr=%s state=%s exit=%s', corr, rec['terminal_state'], rc)
    return result


def return_result(cfg, rec):
    result_path = Path(rec.get('result_path', ''))
    if not result_path.is_file():
        result_path = render_result(cfg, rec)
    try:
        cp = subprocess.run(
            [cfg['send_helper'], str(result_path), rec['correlation_id']],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=60, check=False,
        )
        rec['return_attempts'] = int(rec.get('return_attempts', 0)) + 1
        rec['last_return_attempt_at'] = now_utc()
        if cp.returncode == 0:
            rec['state'] = 'RETURNED'
            rec['returned_at'] = now_utc()
            rec['return_status'] = 'PASS'
            write_state(cfg, rec)
            log.info('returned corr=%s result_sha256=%s', rec['correlation_id'], rec.get('result_sha256'))
            return True
        rec['return_status'] = f'FAILED:{cp.returncode}'
        write_state(cfg, rec)
        log.error('return_failed corr=%s rc=%s', rec['correlation_id'], cp.returncode)
    except Exception as exc:
        rec['return_attempts'] = int(rec.get('return_attempts', 0)) + 1
        rec['last_return_attempt_at'] = now_utc()
        rec['return_status'] = f'ERROR:{type(exc).__name__}'
        write_state(cfg, rec)
        log.error('return_error corr=%s type=%s', rec['correlation_id'], type(exc).__name__)
    return False


def recover_states(cfg):
    tasks = Path(cfg['state_dir']) / 'tasks'
    for p in sorted(tasks.glob('*.json')):
        try:
            rec = load_json(p)
        except Exception:
            log.error('state_unreadable file=%s', p.name)
            continue
        if rec.get('state') == 'RUNNING':
            rec['state'] = rec['terminal_state'] = 'FAILED'
            rec['finished_at'] = now_utc()
            rec['exit_code'] = None
            rec['failure_reason'] = 'runner_restart_while_running'
            write_state(cfg, rec)
            render_result(cfg, rec)
            log.warning('recovered_running_as_failed corr=%s', rec.get('correlation_id'))


def process_auth(cfg, auth_path):
    m = AUTH_RE.fullmatch(auth_path.name)
    if not m:
        return
    corr = m.group(1)
    auth_receipt = receipt_for(auth_path)
    if not auth_receipt.exists():
        return
    rec = read_state(cfg, corr)
    if rec is None:
        rec = {
            'correlation_id': corr,
            'task_id': corr,
            'state': 'RECEIVED',
            'received_at': now_utc(),
            'execution_count': 0,
            'return_attempts': 0,
        }
        write_state(cfg, rec)
        log.info('received corr=%s auth=%s', corr, auth_path.name)
    state = rec.get('state')
    if state == 'RETURNED' or state == 'REJECTED':
        return
    if state in TERMINAL:
        return_result(cfg, rec)
        return
    if state == 'RUNNING':
        return
    try:
        payload = validate_envelope(cfg, auth_path, corr, rec)
        if payload is None:
            return
        rec = read_state(cfg, corr)
        result = run_task(cfg, rec, payload)
        rec = read_state(cfg, corr)
        return_result(cfg, rec)
    except FileNotFoundError:
        return
    except Exception as exc:
        rec = read_state(cfg, corr) or rec
        if rec.get('state') in {'RUNNING'} | TERMINAL:
            rec['failure_reason'] = f'runner_error:{type(exc).__name__}'
            if rec.get('state') == 'RUNNING':
                rec['state'] = rec['terminal_state'] = 'FAILED'
                rec['finished_at'] = now_utc()
                rec['exit_code'] = None
                write_state(cfg, rec)
                render_result(cfg, rec)
                return_result(cfg, rec)
            return
        reject(cfg, rec, f'{type(exc).__name__}:{exc}')


def main():
    cfg = load_json(CONFIG_PATH)
    Path(cfg['state_dir']).joinpath('tasks').mkdir(parents=True, exist_ok=True)
    recover_states(cfg)
    log.info('started inbox=%s user=%s auto_transport_accept_execute=no', cfg['inbox_dir'], cfg['execution_user'])
    while True:
        try:
            for auth in sorted(Path(cfg['inbox_dir']).glob('KVEN-QWENII-AUTH-*.json')):
                process_auth(cfg, auth)
        except Exception as exc:
            log.error('loop_error=%s', type(exc).__name__)
        time.sleep(int(cfg['poll_seconds']))


if __name__ == '__main__':
    main()
