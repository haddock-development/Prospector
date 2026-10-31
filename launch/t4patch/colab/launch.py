import subprocess, os
subprocess.Popen(["nohup", "python3", "/content/job.py"],
                 stdout=open("/content/nohup.out", "w"),
                 stderr=subprocess.STDOUT, start_new_session=True)
print("LAUNCHED")
