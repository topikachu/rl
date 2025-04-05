package com.opentext.sma.robocode.robot;

import com.google.protobuf.Empty;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import robocode.AdvancedRobot;
import robocode.BulletHitBulletEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;
import robocode.SkippedTurnEvent;
import robocode.WinEvent;
import robot.Robot;
import robot.RobotServiceGrpc;

import java.util.concurrent.TimeUnit;

@Slf4j
public class NeuralRobot extends AdvancedRobot {

    static final Empty EMPTY = Empty.newBuilder().build();
    private static final double WALL_THRESHOLD = 50;
    private static final String PYTHON_SERVER_HOST = "localhost";
    private static final int PYTHON_SERVER_PORT = 5001;
    public static final RobotMapper ROBOT_MAPPER = RobotMapper.INSTANCE;
    private static final long UPDATE_INTERVAL_TURNS = 100; // turns

    private RobotServiceGrpc.RobotServiceBlockingStub blockingStub;
    private ManagedChannel channel;

    private int skippedTurns = 0;
    private long lastUpdateTime = 0;

    @Override
    public void run() {
        log.debug("NeuralRobot starting up");
        initializeGrpcConnection();
        initializeRobot();

        startRound();


        while (true) {
            setTurnRadarRight(360);

            // Check if it's time to send an update
            long currentTime = getTime();
            if (currentTime - lastUpdateTime >= UPDATE_INTERVAL_TURNS) {
                sendStateToPythonAndPerformAction(null);
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
        endRound(Robot.RoundResult.Reason.UNKNOWN);
        log.debug("Robot initialization complete");
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {

        log.debug("Sending state to Python server at onScannedRobot");
        sendStateToPythonAndPerformAction(event);

    }





    private void performActions(Robot.Actions actions) {
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

    @Override
    public void onBulletHit(BulletHitEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setBulletHit(ROBOT_MAPPER.bulletHitToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onBulletHitBullet(BulletHitBulletEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setBulletHitBullet(ROBOT_MAPPER.bulletHitBulletToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onBulletMissed(BulletMissedEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setBulletMissed(ROBOT_MAPPER.bulletMissedToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setHitByBullet(ROBOT_MAPPER.hitByBulletToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setHitRobot(ROBOT_MAPPER.hitRobotToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setHitWall(ROBOT_MAPPER.hitWallToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

    @Override
    public void onWin(WinEvent event) {
        endRound(Robot.RoundResult.Reason.WIN);
    }

    @Override
    public void onDeath(DeathEvent event) {
        endRound(Robot.RoundResult.Reason.LOSS);
        log.info("Round lost.");
    }

    @Override
    public void onSkippedTurn(SkippedTurnEvent event) {
        skippedTurns++;
        log.info("Skipped turn: {}", skippedTurns);
    }



    private void sendEventToPython(Robot.Event event) {
        try {
            blockingStub.onEvent(event);
            log.debug("Sent event to Python server: {}", event.getEventTypeCase());
        } catch (StatusRuntimeException e) {
            log.error("gRPC error when sending event to Python: {} - {}", e.getStatus(), e.getMessage());
        } catch (Exception e) {
            log.error("Unexpected error when sending event to Python: {}", e.getMessage(), e);
        }
    }


    private void sendStateToPythonAndPerformAction(ScannedRobotEvent enemy) {
        try {
            lastUpdateTime = getTime();
            Robot.Actions actions = blockingStub.sendState(ROBOT_MAPPER.gameStateToProto(this, enemy));
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

    private void endRound(Robot.RoundResult.Reason reason) {
        try {
            log.debug("Ending round");
            Robot.RoundResult result = Robot.RoundResult.newBuilder().setReason(reason).build(); // Add more fields as needed (e.g., score, rank, etc.)
//                            .
            blockingStub.endRound(result);
            log.debug("Round end signal sent to Python server");
        } catch (StatusRuntimeException e) {
            log.error("Error ending round: {}", e.getMessage(), e);
        }
    }


    private void startRound() {
        try {
            log.debug("Start a round");

            blockingStub.startRound(Empty.newBuilder().build());
            log.debug("Round start signal sent to Python server");
        } catch (StatusRuntimeException e) {
            log.error("Error start round: {}", e.getMessage(), e);
        }
    }

    private void cleanupGrpcConnection() {
        if (channel != null && !channel.isShutdown()) {
            try {
                channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                log.error("Error shutting down gRPC channel: {}", e.getMessage(), e);
                Thread.currentThread().interrupt();
            } finally {
                if (!channel.isShutdown()) {
                    channel.shutdownNow();
                }
            }
            log.debug("gRPC connection cleaned up");
        }
    }


    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        Robot.Event rpcEvent = Robot.Event.newBuilder().setRobotDeath(ROBOT_MAPPER.robotDeathToProto(event)).build();
        sendEventToPython(rpcEvent);
    }

}
