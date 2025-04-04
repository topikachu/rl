package com.opentext.sma.robocode.robot;

import org.mapstruct.Builder;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.factory.Mappers;
import robocode.AdvancedRobot;
import robocode.ScannedRobotEvent;
import robot.Robot;

@Mapper

public interface RobotMapper {
    RobotMapper INSTANCE = Mappers.getMapper( RobotMapper.class );

    Robot.RobotState robotToState(AdvancedRobot robot);
    Robot.EnemyState enemyToState(ScannedRobotEvent enemy);
}
