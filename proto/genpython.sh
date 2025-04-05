python -m grpc_tools.protoc -I. \
    --python_betterproto_out=../neural \
    robot.proto

