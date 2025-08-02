import subprocess
#
cmd = ["python", "main.py", "--epoch", '200']
print(f"\n>>> Running command: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # print(">>> STDOUT:\n", result.stdout.strip())
except subprocess.CalledProcessError as e:
    print(">>> ERROR occurred:")
    print(e.stderr.strip())

cmd = ["python", "main_text_encoder.py", "--epoch", '300']
print(f"\n>>> Running command: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # print(">>> STDOUT:\n", result.stdout.strip())
except subprocess.CalledProcessError as e:
    print(">>> ERROR occurred:")
    print(e.stderr.strip())


cmd = ["python", "metrics_single.py"]
print(f"\n>>> Running command: {' '.join(cmd)}")
try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(">>> STDOUT:\n", result.stdout.strip())
except subprocess.CalledProcessError as e:
    print(">>> ERROR occurred:")
    print(e.stderr.strip())