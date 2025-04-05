package com.opentext.sma.robocode.robot;

import com.google.protobuf.Empty;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import robocode.AdvancedRobot;
import robocode.ScannedRobotEvent;
import robot.Robot;
import robot.RobotServiceGrpc;

@Slf4j
public   class ActRobot extends AdvancedRobot {

    private static final String PYTHON_SERVER_HOST = "localhost";
    private static final int PYTHON_SERVER_PORT = 5001;
    public static final RobotMapper ROBOT_MAPPER = RobotMapper.INSTANCE;
    private static final long UPDATE_INTERVAL_TURNS = 100; // turns

    private RobotServiceGrpc.RobotServiceBlockingStub blockingStub;
    private ManagedChannel channel;

    private long lastUpdateTime = 0;

    @Override
    public void run() {
        log.debug("NeuralRobot starting up");
        initializeGrpcConnection();
        initializeRobot();
        while (true) {
            setTurnRadarRight(360);
            // Check if it's time to send an update
            long currentTime = getTime();
            if (currentTime - lastUpdateTime >= UPDATE_INTERVAL_TURNS) {
                act(null);
            }

            execute();
        }
    }

    private void initializeGrpcConnection() {
        log.debug("Initializing gRPC connection to {}:{}", PYTHON_SERVER_HOST, PYTHON_SERVER_PORT);
        if (channel == null || channel.isShutdown()) {
            channel = ManagedChannelBuilder.forAddress(PYTHON_SERVER_HOST, PYTHON_SERVER_PORT)
                    .usePlaintext()
                    .build();
            blockingStub = RobotServiceGrpc.newBlockingStub(channel);
            log.debug("gRPC connection initialized");
        } else {
            log.debug("Using existing gRPC connection");
        }
    }

    private void initializeRobot() {
        log.debug("Initializing robot settings");
        setAdjustRadarForGunTurn(true);
        setAdjustGunForRobotTurn(true);
        log.debug("Robot initialization complete");
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {

        log.debug("Sending state to Python server at onScannedRobot");
        act(event);

    }





    protected void performActions(Robot.Actions actions) {
        log.debug("Performing {} actions", actions.getActionsCount());
        for (Robot.Action action : actions.getActionsList()) {
            executeAction(action);
        }
    }

    private void executeAction(Robot.Action action) {
        double value = action.getValue();
        log.debug("Executing action: {}, value: {}", action.getActionType(), value);

        switch (action.getActionType()) {
            case MOVE_FORWARD: setAhead(value); break;
            case MOVE_BACKWARD: setBack(value); break;
            case TURN_LEFT: setTurnLeft(value); break;
            case TURN_RIGHT: setTurnRight(value); break;
            case TURN_GUN_LEFT: setTurnGunLeft(value); break;
            case TURN_GUN_RIGHT: setTurnGunRight(value); break;
            case FIRE: setFire(Math.min(3, Math.max(0.1, value))); break;
            case DO_NOTHING:
                log.debug("Doing nothing");
                break;
        }
        execute();
    }















    private void act(ScannedRobotEvent enemy) {
        try {
            lastUpdateTime = getTime();
            Robot.Actions actions = blockingStub.act(ROBOT_MAPPER.gameStateToProto(this, enemy));
            log.debug("Received actions from Python server: {}", actions);
            performActions(actions);
        } catch (StatusRuntimeException e) {
            log.error("gRPC error when sending state to Python: {} - {}", e.getStatus(), e.getMessage());
            // Optionally, you might want to attempt reconnecting to the gRPC server here
            // initializeGrpcConnection();
        } catch (Exception e) {
            log.error("Unexpected error when sending state to Python: {}", e.getMessage(), e);
        }
    }










}
