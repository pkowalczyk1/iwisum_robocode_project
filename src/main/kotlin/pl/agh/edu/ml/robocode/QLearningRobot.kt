package pl.agh.edu.ml.robocode

import robocode.AdvancedRobot
import robocode.BulletHitEvent
import robocode.BulletMissedEvent
import robocode.HitByBulletEvent
import robocode.HitRobotEvent
import robocode.HitWallEvent
import robocode.RobotDeathEvent
import robocode.RoundEndedEvent
import robocode.ScannedRobotEvent
import java.lang.Math.toDegrees
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class QLearningRobot : AdvancedRobot() {

    private val driver = TcpRobotDriver()

    private var observedEnergy: Double = 0.0
    private var shotHit: Boolean = false
    private var shotMissed: Boolean = false
    private var robotCollision: Boolean = false
    private var wallCollision: Boolean = false
    private var robotHit: Boolean = false
    private var dead: Boolean = false

    private var lastEnemyPosition: Position? = null

    override fun run() {
        try {
            driver.sendResult(getResult(Action.MOVE_FORWARD))
            while (true) {
                val action = driver.getAction()
                doAction(action)
                driver.sendResult(getResult(action))
                reset()
            }
        } catch (e: Exception) {
            println("Disconnected from server or round ended: ${e.message}")
        }
    }

    fun getResult(action: Action): Result {
        val observation = Observation(
            action = action.ordinal,
            xPos = x,
            yPos = y,
            energy = energy,
            velocity = velocity,
            enemyDistance = lastEnemyPosition?.distance(Position(x, y)),
            enemyHeading = lastEnemyPosition?.heading(Position(x, y)),
            wallHit = if (wallCollision) 1 else 0,
            robotHit = if (robotHit) 1 else 0
        )
        return Result(
            observation = observation,
            reward = getReward()
        )
    }

    override fun onScannedRobot(event: ScannedRobotEvent?) {
        if (event == null) {
            return
        }

        fire(5.0)
        val enemyDistance = event.distance
        val enemyBearing = event.bearing
        lastEnemyPosition = Position(x + enemyDistance * sin(Math.toRadians(heading + enemyBearing)),
            y + enemyDistance * cos(Math.toRadians(heading + enemyBearing)))
    }

    override fun onBulletHit(event: BulletHitEvent?) {
        shotHit = true
    }

    override fun onBulletMissed(event: BulletMissedEvent?) {
        shotMissed = true
    }

    override fun onHitByBullet(event: HitByBulletEvent?) {
        robotHit = true
    }

    override fun onHitWall(event: HitWallEvent?) {
        wallCollision = true
    }

    override fun onRobotDeath(event: RobotDeathEvent?) {
        dead = true
    }

    override fun onHitRobot(event: HitRobotEvent?) {
        robotCollision = true
        if (event != null && event.bearing < 10 && event.bearing > -10) {
            fire(2.0)
        }

        if (event != null && event.isMyFault) {
            back(30.0)
        }
    }

    override fun onRoundEnded(event: RoundEndedEvent?) {
        driver.closeConnection()
    }

    private fun doAction(action: Action) {
        when (action) {
            Action.MOVE_FORWARD -> ahead(10.0)
            Action.MOVE_BACKWARD -> back(10.0)
            Action.ROTATE_LEFT -> turnLeft(10.0)
            Action.ROTATE_RIGHT -> turnRight(10.0)
        }
    }

    private fun getReward(): Double {
        val shotHitReward = if (shotHit) 10.0 else 0.0
        val shotMissedReward = if (shotMissed) -1.0 else 0.0
        val robotHitReward = if (robotHit) -5.0 else 0.0
        val wallCollisionReward = if (wallCollision) -3.0 else 0.0
        val robotCollisionReward = if (robotCollision) -3.0 else 0.0
        val deadReward = if (dead) -10.0 else 0.0
        val energyReward = (observedEnergy - energy) * 0.03
        return shotHitReward +
            shotMissedReward +
            robotHitReward +
            wallCollisionReward +
            robotCollisionReward +
            deadReward +
            energyReward
    }

    private fun reset() {
        observedEnergy = energy
        shotHit = false
        shotMissed = false
        robotCollision = false
        wallCollision = false
        robotHit = false
        dead = false
    }
}

data class Position(val x: Double, val y: Double) {

    fun distance(other: Position): Double {
        return sqrt((other.x - x).pow(2.0) + (other.y - y).pow(2.0))
    }

    fun heading(other: Position): Double {
        val dx = other.x - x
        val dy = other.y - y

        var angle = toDegrees(atan2(dx, dy))

        if (angle < 0) {
            angle += 360
        }

        return angle
    }
}
