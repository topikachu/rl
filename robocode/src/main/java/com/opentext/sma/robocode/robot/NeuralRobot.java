package com.opentext.sma.robocode.robot;

import com.google.protobuf.Empty;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import robocode.*;
import robocode.util.Utils;
import robot.Robot;
import robot.RobotServiceGrpc;

import java.util.concurrent.TimeUnit;

@Slf4j
public class NeuralRobot extends AdvancedRobot {

    static final Empty EMPTY = Empty.newBuilder().build();
    private static final double WALL_THRESHOLD = 50;
    private static final String PYTHON_SERVER_HOST = "localhost";
    private static final int PYTHON_SERVER_PORT = 5000;

    private RobotServiceGrpc.RobotServiceBlockingStub blockingStub;
    private ManagedChannel channel;
    private Robot.RoundResult.Result roundResult;

    private int skippedTurns = 0;

    @Override
    public void run() {
        log.debug("NeuralRobot starting up");
        initializeGrpcConnection();
        initializeRobot();

        startRound();


        while (true) {
            setTurnRadarRight(360);
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
        roundResult = Robot.RoundResult.Result.ROUND_END;
        endRound(Robot.RoundResult.Result.ROUND_END);
        log.debug("Robot initialization complete");
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent e) {
        log.debug("Robot scanned");
        Robot.RobotState robotState = buildRobotState(e);
        Robot.EnemyState enemyState = RobotMapper.INSTANCE.enemyToState(e);


        try {
            log.debug("Sending state to Python server at onScannedRobot");
            Robot.Actions actions = sendStateToPython(robotState, enemyState);
            log.debug("Received actions from Python server: {} at onScannedRobot", actions);
            performActions(actions);
        } catch (Exception ex) {
            log.error("Error sending state to Python: {}", ex.getMessage(), ex);
        }
    }

    private Robot.RobotState buildRobotState(ScannedRobotEvent e) {
        RobotMapper mapper = RobotMapper.INSTANCE;
        Robot.RobotState robotState = mapper.robotToState(this);

        int nearWall = isNearWall() ? 1 : 0;
        double bearingFromGun = calculateBearingFromGun(e);

        log.debug("Near wall: {}, Bearing from gun: {}", nearWall, bearingFromGun);

        return robotState.toBuilder()
                .setNearWall(nearWall)
                .setGunBearing(bearingFromGun)
                .build();
    }

    private boolean isNearWall() {
        double distLeftWall = getX();
        double distRightWall = getBattleFieldWidth() - getX();
        double distTopWall = getBattleFieldHeight() - getY();
        double distBottomWall = getY();

        boolean near = distLeftWall < WALL_THRESHOLD || distRightWall < WALL_THRESHOLD ||
                distTopWall < WALL_THRESHOLD || distBottomWall < WALL_THRESHOLD;

        log.debug("Wall distances - Left: {}, Right: {}, Top: {}, Bottom: {}",
                distLeftWall, distRightWall, distTopWall, distBottomWall);

        return near;
    }

    private double calculateBearingFromGun(ScannedRobotEvent e) {
        double absoluteBearing = getHeading() + e.getBearing();
        double bearingFromGun = Utils.normalRelativeAngleDegrees(absoluteBearing - getGunHeading());
        log.debug("Absolute bearing: {}, Bearing from gun: {}", absoluteBearing, bearingFromGun);
        return bearingFromGun;
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
    public void onWin(WinEvent event) {
        log.info("Round won!");
        this.roundResult = Robot.RoundResult.Result.WIN;
    }

    @Override
    public void onDeath(DeathEvent event) {
        log.info("Round lost.");
        this.roundResult = Robot.RoundResult.Result.LOSS;
    }

    @Override
    public void onSkippedTurn(SkippedTurnEvent event) {
        skippedTurns++;
        log.info("Skipped turn: {}", skippedTurns);
    }

    @Override
    public void onRoundEnded(RoundEndedEvent event) {
        log.info("Round ended. Result: {}", this.roundResult);
        endRound(this.roundResult);
        cleanupGrpcConnection();
    }


    private Robot.Actions sendStateToPython(Robot.RobotState robotState, Robot.EnemyState enemyState) {
        Robot.GameState gameState = Robot.GameState.newBuilder()
                .setRobotState(robotState)
                .setEnemyState(enemyState)
                .build();
        Robot.Actions actions = blockingStub.sendState(gameState);
        return actions;
    }

    private void endRound(Robot.RoundResult.Result result) {
        try {
            log.debug("Ending round with result: {}", result);
            Robot.RoundResult roundResult = Robot.RoundResult.newBuilder()
                    .setResult(result)
                    .build();
            blockingStub.endRound(roundResult);
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

}
