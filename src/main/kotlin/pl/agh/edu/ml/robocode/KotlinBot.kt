package pl.agh.edu.ml.robocode

import robocode.AdvancedRobot
import robocode.ScannedRobotEvent
import robocode.HitByBulletEvent
import robocode.HitWallEvent
import java.awt.Color

class KotlinBot : AdvancedRobot() {

    override fun run() {
        // Set colors
        setBodyColor(Color(0, 128, 255))
        setGunColor(Color(255, 255, 255))
        setRadarColor(Color(255, 0, 0))
        setBulletColor(Color(255, 255, 0))

        // Let the radar turn independently
        isAdjustRadarForGunTurn = true
        // Let the gun turn independently
        isAdjustGunForRobotTurn = true

        // Main loop: spin radar to look for enemies
        while (true) {
            turnRadarRight(360.0)  // continuous 360° radar sweep
            execute()
        }
    }

    override fun onScannedRobot(event: ScannedRobotEvent) {
        // Calculate absolute bearing to the scanned robot
        val absoluteBearing = heading + event.bearing

        // Calculate gun turn angle (normalize to -180..180)
        val bearingFromGun = normalizeBearing(absoluteBearing - gunHeading)

        // Turn gun toward the target
        turnGunRight(bearingFromGun)

        // Fire with power depending on distance (closer = more power)
        val firePower = when {
            event.distance < 150 -> 3.0
            event.distance < 300 -> 2.0
            else -> 1.0
        }
        if (Math.abs(gunTurnRemaining) < 10) {
            fire(firePower)
        }

        // Lock radar on the target by turning it slightly extra
        val bearingFromRadar = normalizeBearing(absoluteBearing - radarHeading)
        turnRadarRight(bearingFromRadar * 1.1)
    }

    override fun onHitByBullet(event: HitByBulletEvent) {
        // Move perpendicular to the bullet to evade
        turnRight(normalizeBearing(90 - (heading - event.heading)))
        ahead(100.0)
    }

    override fun onHitWall(event: HitWallEvent) {
        // Bounce off the wall
        back(50.0)
        turnRight(90.0)
        ahead(100.0)
    }

    /**
     * Normalizes a bearing to the range [-180, 180]
     */
    private fun normalizeBearing(angle: Double): Double {
        var a = angle
        while (a > 180) a -= 360
        while (a < -180) a += 360
        return a
    }
}
