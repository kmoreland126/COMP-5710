'''
Author: Akond Rahman 
'''

import random 
import string
import traceback
import time

def divide(v1, v2):
    return v1/ v2 

def simpleFuzzer(): 
    """Lightweight fuzzer that generates random alphanumeric inputs and calls
    divide(v1, v2).

    It keeps trying until it records 10 exceptions (or reaches a practical
    attempt limit), then writes those error reports to 'errors.txt' in the same
    folder as this script. Each report includes the inputs that caused the
    failure, the exception type/message, and the full traceback to help with
    debugging.
    """

    errors = []
    attempts = 0
    max_attempts = 100000

    # Keep trying until we've captured 10 exceptions or we've made enough attempts.
    while len(errors) < 10 and attempts < max_attempts:
        attempts += 1
        # Create two random alphanumeric strings (length 1–12 chars)
        len1 = random.randint(1, 12)
        len2 = random.randint(1, 12)
        s1 = ''.join(random.choices(string.ascii_letters + string.digits, k=len1))
        s2 = ''.join(random.choices(string.ascii_letters + string.digits, k=len2))

        try:
            _ = divide(s1, s2)
        except Exception as e:
            tb = traceback.format_exc()
            record = {
                'inputs': (s1, s2),
                'exception_type': type(e).__name__,
                'message': str(e),
                'traceback': tb,
                'attempt': attempts,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            errors.append(record)

    # Save the collected error reports to 'errors.txt'.
    out_path = 'errors.txt'
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            for idx, rec in enumerate(errors, start=1):
                fh.write(f"Error #{idx}\n")
                fh.write(f"Timestamp: {rec['timestamp']}\n")
                fh.write(f"Attempt: {rec['attempt']}\n")
                fh.write(f"Inputs: {rec['inputs']}\n")
                fh.write(f"Exception: {rec['exception_type']}: {rec['message']}\n")
                fh.write("Traceback:\n")
                fh.write(rec['traceback'])
                fh.write('\n' + ('-'*60) + '\n')
        print(f"Collected {len(errors)} errors and wrote them to {out_path}")
    except Exception as e:
        print(f"Failed to write errors to {out_path}: {e}")


if __name__=='__main__':
    simpleFuzzer()
