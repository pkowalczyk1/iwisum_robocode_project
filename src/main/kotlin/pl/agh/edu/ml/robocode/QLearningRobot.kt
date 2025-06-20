package pl.agh.edu.ml.robocode

import robocode.AdvancedRobot
import robocode.BattleEndedEvent
import robocode.BulletHitEvent
import robocode.BulletMissedEvent
import robocode.HitByBulletEvent
import robocode.HitRobotEvent
import robocode.HitWallEvent
import robocode.RobotDeathEvent
import robocode.RoundEndedEvent
import robocode.ScannedRobotEvent
import robocode.WinEvent
import java.lang.Math.toDegrees
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class QLearningRobot : AdvancedRobot() {

    companion object {
        private const val Y_MAX = 600
        private const val X_MAX = 800
    }

    private val driver = TcpRobotDriver()

    private var observedEnergy: Double = 0.0
    private var shotHit: Boolean = false
    private var shotMissed: Boolean = false
    private var robotCollision: Boolean = false
    private var wallCollision: Boolean = false
    private var robotHit: Boolean = false
    private var dead: Boolean = false
    private var battleEnded: Boolean = false
    private var winner: Boolean = false

    private var lastEnemyPosition: Position? = null

    override fun run() {
        try {
            driver.sendResult(getResult())
            while (true) {
                val action = driver.getAction()
                doAction(action)
                driver.sendResult(getResult())
                reset()
            }
        } catch (e: Exception) {
            println("Disconnected from server or round ended: ${e.message}")
        }
    }

    fun getResult(): Result {
        val observation = Observation(
            xPos = x,
            yPos = y,
            heading = heading,
            energy = energy,
            enemyDistance = lastEnemyPosition?.distance(Position(x, y)),
            enemyHeading = lastEnemyPosition?.heading(Position(x, y)),
            wallHit = if (wallCollision) 1 else 0,
            robotHit = if (robotHit) 1 else 0,
            done = battleEnded
        )
        return Result(
            observation = observation,
            reward = getReward()
        )
    }

    override fun onBattleEnded(event: BattleEndedEvent?) {
        battleEnded = true
    }

    override fun onWin(event: WinEvent?) {
        winner = true
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
//            Action.MOVE_FORWARD -> ahead(20.0)
            Action.MOVE_FAST_FORWARD -> ahead(40.0)
//            Action.MOVE_BACKWARD -> back(20.0)
            Action.MOVE_FAST_BACKWARD -> back(40.0)
//            Action.ROTATE_LEFT -> turnLeft(20.0)
            Action.ROTATE_MORE_LEFT -> turnLeft(50.0)
//            Action.ROTATE_RIGHT -> turnRight(20.0)
            Action.ROTATE_MORE_RIGHT -> turnRight(50.0)
            Action.SCAN -> turnGunRight(360.0)
        }
    }

    private fun getReward(): Double {
        val shotHitReward = if (shotHit) 5.0 else 0.0
//        val shotMissedReward = if (shotMissed) -1.0 else 0.0
        val robotHitReward = if (robotHit) -7.0 else 0.0
        val wallCollisionReward = if (wallCollision) -5.0 else 0.0
        val robotCollisionReward = if (robotCollision) -3.0 else 0.0
        val winReward = if (winner) 20.0 else 0.0
        val energyReward = (observedEnergy - energy) * 0.03
        return shotHitReward +
//            shotMissedReward +
            winReward +
            robotHitReward +
            wallCollisionReward +
            robotCollisionReward +
            energyReward +
            getWallProximityPenalty()
    }

    private fun getWallDistance(): Double {
        val yDistance = min(y, Y_MAX - y)
        val xDistance = min(x, X_MAX - x)
        return min(yDistance, xDistance)
    }

    private fun getWallProximityPenalty(): Double {
        val distance = getWallDistance()
        return if (distance < 50.0) {
            -((50.0 - distance) / 5.0)
        } else {
            0.0
        }
    }

    private fun reset() {
        observedEnergy = energy
        shotHit = false
        shotMissed = false
        robotCollision = false
        wallCollision = false
        robotHit = false
        dead = false
        battleEnded = false
        winner = false
    }
}

data class Position(val x: Double, val y: Double) {

    fun distance(other: Position): Double {
        return sqrt((other.x - x).pow(2.0) + (other.y - y).pow(2.0))
    }

    fun heading(other: Position): Double {
        val dx = other.x - x
        val dy = other.y - y
        val angleRadians = atan2(dx, dy)
        val angleDegrees = toDegrees(angleRadians)
        return (angleDegrees + 360) % 360
    }
}
