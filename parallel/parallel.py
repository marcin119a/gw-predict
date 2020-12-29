import subprocess


mongo_cmd = "setsid parallel/mongo.sh"
process = subprocess.Popen(mongo_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(error)

worker_cmd = "setsid parallel/worker.sh"
process = subprocess.Popen(mongo_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
print(error)

template = 'python lstm.py'

args = [[], []]

# Run commands in parallel
processes = []

for arg in args:
    command = template.format(*[str(a) for a in arg])
    process = subprocess.Popen(command, shell=True)
    processes.append(process)

# Collect statuses
output = [p.wait() for p in processes]
