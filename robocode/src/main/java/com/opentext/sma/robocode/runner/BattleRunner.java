package com.opentext.sma.robocode.runner;

import com.opentext.sma.robocode.robot.NeuralRobot;
import com.opentext.sma.robocode.robot.SampleRamFire;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import robocode.control.BattleSpecification;
import robocode.control.BattlefieldSpecification;
import robocode.control.RobocodeEngine;
import robocode.control.RobotSpecification;
import robocode.control.events.BattleAdaptor;
import robocode.control.events.BattleCompletedEvent;
import robocode.control.events.BattleErrorEvent;
import robocode.control.events.BattleMessageEvent;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j
public class BattleRunner {
    public static void main(String[] args) {
        log.info("Starting Robocode Battle");

        RobocodeEngine.setLogMessagesEnabled(false);
        RobocodeEngine engine = new RobocodeEngine(new File("."));

        engine.addBattleListener(new BattleObserver());
        boolean visible = Boolean.parseBoolean(System.getProperty("VISIBLE", "false"));
        engine.setVisible(visible);

        int numberOfRounds = 5;
        BattlefieldSpecification battlefield = new BattlefieldSpecification(400, 400);
        String neuralRobotName = NeuralRobot.class.getName() + "*";

        RobotSpecification neuralRobot = engine.getLocalRepository(neuralRobotName)[0];


        List<String> enemyRobots = Arrays.asList(
                "sample.Target",
                "sample.TrackFire",
                "sample.RamFire",
                "sample.Walls",
                "sample.SpinBot"
        );

        while (true) {
            Random random = new Random();
            String selectedEnemyName = enemyRobots.get(random.nextInt(enemyRobots.size()));
            RobotSpecification enemy = engine.getLocalRepository(selectedEnemyName)[0];
            RobotSpecification[] robots = new RobotSpecification[]{neuralRobot, enemy};

            BattleSpecification battleSpec = new BattleSpecification(numberOfRounds, battlefield,robots);
            engine.runBattle(battleSpec, true);
        }
    }

    static class BattleObserver extends BattleAdaptor {
        private static final Logger logger = LoggerFactory.getLogger(BattleObserver.class);

        public void onBattleCompleted(BattleCompletedEvent e) {
            logger.info("-- Battle has completed --");
            logger.info("Battle results:");
            for (robocode.BattleResults result : e.getSortedResults()) {
                logger.info("  {}: {}", result.getTeamLeaderName(), result.getScore());
            }
        }

        public void onBattleMessage(BattleMessageEvent e) {
            logger.info("Msg> {}", e.getMessage());
        }

        public void onBattleError(BattleErrorEvent e) {
            logger.error("Err> {}", e.getError());
        }
    }
}
