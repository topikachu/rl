package com.opentext.sma.robocode.robot;

import org.mapstruct.Mapper;
import org.mapstruct.factory.Mappers;
import robocode.*;
import robot.Robot;

@Mapper
public interface RobotMapper {
    RobotMapper INSTANCE = Mappers.getMapper(RobotMapper.class);

    Robot.RobotState robotStateToProto(AdvancedRobot robot);
    Robot.ScannedRobotEvent scannedRobotToProto(ScannedRobotEvent enemy);
    Robot.BulletHitEvent bulletHitToProto(BulletHitEvent e);
    Robot.BulletHitBulletEvent bulletHitBulletToProto(BulletHitBulletEvent e);
    Robot.BulletMissedEvent bulletMissedToProto(BulletMissedEvent e);
    Robot.HitByBulletEvent hitByBulletToProto(HitByBulletEvent e);
    Robot.HitRobotEvent hitRobotToProto(HitRobotEvent e);
    Robot.HitWallEvent hitWallToProto(HitWallEvent e);
    Robot.RobotDeathEvent robotDeathToProto(RobotDeathEvent e);
    Robot.Bullet bulletToProto(Bullet bullet);
    Robot.GameState gameStateToProto(AdvancedRobot robotState, ScannedRobotEvent enemy);

}
