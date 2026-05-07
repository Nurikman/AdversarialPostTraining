from openai import OpenAI
client = OpenAI()
for job in client.fine_tuning.jobs.list():
    print(job.id, job.fine_tuned_model)