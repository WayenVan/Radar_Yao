import sys

sys.path.append("src")


from radar.modeling_diff.diff_pipline import RDDPMPipeline

pipe = RDDPMPipeline.from_pretrained("outputs/rddpm-pipeline-test")
print(pipe.unet)
