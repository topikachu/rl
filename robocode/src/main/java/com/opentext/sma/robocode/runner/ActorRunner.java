package com.opentext.sma.robocode.runner;

import com.opentext.sma.robocode.robot.ActRobot;
import com.opentext.sma.robocode.robot.NeuralRobot;
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
public class ActorRunner {
    public static void main(String[] args) {
        log.info("Starting Robocode Battle");

        RobocodeEngine.setLogMessagesEnabled(false);
        RobocodeEngine engine = new RobocodeEngine(new File("."));

        engine.setVisible(true);

        int numberOfRounds = 1;
        BattlefieldSpecification battlefield = new BattlefieldSpecification(400, 400);
        String actorRobotName = ActRobot.class.getName() + "*";

        RobotSpecification actorRobot = engine.getLocalRepository(actorRobotName)[0];


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
            RobotSpecification[] robots = new RobotSpecification[]{actorRobot, enemy};

            BattleSpecification battleSpec = new BattleSpecification(numberOfRounds, battlefield,robots);
            engine.runBattle(battleSpec, true);
        }
    }

}
