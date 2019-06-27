python /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/mo.py \
  --input_model=inception_v3_sdd_bs50_lr0.01_ep0.onnx \
  --disable_fusing --disable_gfusing \
  --reverse_input_channels
