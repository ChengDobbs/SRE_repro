import wandb
api = wandb.Api()
run = api.run("jch26/SRE2L3.0/run_id")
run.group = "new_group_name"
run.update()