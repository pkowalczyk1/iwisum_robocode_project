package pl.agh.edu.ml.robocode

import kotlinx.serialization.Serializable

@Serializable
data class Observation(
    val xPos: Double,
    val yPos: Double,
    val energy: Double,
    val heading: Double,
    val enemyDistance: Double?,
    val enemyHeading: Double?,
    val wallHit: Int,
    val robotHit: Int,
    val done: Boolean)
