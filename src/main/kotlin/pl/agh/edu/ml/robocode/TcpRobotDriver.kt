package pl.agh.edu.ml.robocode

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.Socket

class TcpRobotDriver {

    private val socket: Socket = Socket("localhost", 5000)
    private val output: BufferedWriter = BufferedWriter(OutputStreamWriter(socket.getOutputStream()))
    private val input: BufferedReader = BufferedReader(InputStreamReader(socket.getInputStream()))

    fun sendResult(result: Result) {
        val jsonResult = Json.encodeToString(result)
        output.write(jsonResult + "\n")
        output.flush()
    }

    fun getAction(): Action {
        val action = input.readLine()
        return Action.entries[action.toInt()]
    }

    fun closeConnection() {
        output.close()
        input.close()
        socket.close()
    }
}

@Serializable
data class Result(
    val observation: Observation,
    val reward: Double,
)