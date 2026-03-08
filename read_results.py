import json

with open('test_results.json') as f:
    data = json.load(f)

with open('test_summary.txt', 'w') as out:
    for d in data:
        status = 'PASS' if d['success'] else 'FAIL'
        out.write(f"--- {d['mode']} : {status} ---\n")
        if not d['success']:
            # grab the last 20 lines of stderr for context
            lines = d.get('stderr', '').strip().split('\n')
            err_text = '\n'.join(lines[-20:])
            out.write(err_text + "\n\n")
