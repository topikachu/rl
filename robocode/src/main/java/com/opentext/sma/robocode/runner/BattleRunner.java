package com.opentext.sma.robocode.runner;

import com.opentext.sma.robocode.robot.NeuralRobot;
import lombok.extern.slf4j.Slf4j;
import net.sf.robocode.battle.IBattleManager;
import net.sf.robocode.battle.IBattleManagerBase;
import net.sf.robocode.core.Container;
import net.sf.robocode.core.ContainerBase;
import net.sf.robocode.core.RobocodeMainBase;
import net.sf.robocode.settings.ISettingsManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import robocode.control.BattleSpecification;
import robocode.control.BattlefieldSpecification;
import robocode.control.RobocodeEngine;
import robocode.control.RobotSpecification;
import robocode.control.events.*;

import java.io.File;

@Slf4j
public class BattleRunner {
    public static void main(String[] args) {

        log.info("Starting Robocode Battle");


        // Disable log messages from Robocode
        RobocodeEngine.setLogMessagesEnabled(false);

        // Create the RobocodeEngine
        //   RobocodeEngine engine = new RobocodeEngine(); // Run from current working directory
        RobocodeEngine engine = new RobocodeEngine(new File(".")); // Run from C:/Robocode

        // Add our own container to the RobocodeEngine

        // Add our own battle listener to the RobocodeEngine
        engine.addBattleListener(new BattleObserver());
//        ISettingsManager settingsManager = Container.getComponent(ISettingsManager.class);
//        settingsManager.setOptionsDevelopmentPaths();
        boolean visible=Boolean.parseBoolean(System.getProperty("VISIBLE","false"));

        // Show the Robocode battle view
        engine.setVisible(visible);


        // Setup the battle specification

        int numberOfRounds = 5000;
        BattlefieldSpecification battlefield = new BattlefieldSpecification(400, 400); // 800x600
        String neuraRobotName=NeuralRobot.class.getName()+"*";
        RobotSpecification[] selectedRobots = engine.getLocalRepository(neuraRobotName+",sample.SittingDuck");
//        RobotSpecification[] selectedRobots = engine.getLocalRepository("sample.Tracker,sample.Target");



        BattleSpecification battleSpec = new BattleSpecification(numberOfRounds, battlefield, selectedRobots);


        // Run our specified battle and let it run till it is over
        engine.runBattle(battleSpec, true); // waits till the battle finishes

        // Cleanup our RobocodeEngine
        engine.close();

        // Make sure that the Java VM is shut down properly
        System.exit(0);
    }


static class BattleObserver extends BattleAdaptor {
    private static final Logger logger = LoggerFactory.getLogger(BattleObserver.class);

    // Called when the battle is completed successfully with battle results
    public void onBattleCompleted(BattleCompletedEvent e) {
        logger.info("-- Battle has completed --");

        // Log the sorted results with the robot names
        logger.info("Battle results:");
        for (robocode.BattleResults result : e.getSortedResults()) {
            logger.info("  {}: {}", result.getTeamLeaderName(), result.getScore());
        }
    }

    // Called when the game sends out an information message during the battle
    public void onBattleMessage(BattleMessageEvent e) {
        logger.info("Msg> {}", e.getMessage());
    }

    // Called when the game sends out an error message during the battle
    public void onBattleError(BattleErrorEvent e) {
        logger.error("Err> {}", e.getError());
    }
}

}
