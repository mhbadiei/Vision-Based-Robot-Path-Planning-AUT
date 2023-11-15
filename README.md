# Vision-Based Robot Path Planning
In this project, drawing upon knowledge from robotics, neural networks, Internet of Things (IoT), machine learning, and information fusion algorithms, we have implemented an intelligent robot within the Gazebo simulator environment.

This robot receives control commands from a direction detection system based on a convolutional neural network (CNN) model. It seamlessly switches between autonomous and manual modes depending on the presence or absence of obstacles. In autonomous mode, the robot utilizes laser sensors for obstacle detection, whereas in manual mode, it responds to commands received through image analysis from an onboard camera, processed via a CNN.

Furthermore, to enhance obstacle detection and navigation, we have employed various scientific classification techniques to determine obstacle positions and distances based on sensor data. Subsequently, we integrated the MQTT (Message Queuing Telemetry Transport) protocol to expedite information exchange between the direction detection system and the robot. MQTT, known for its speed and suitability for inter-device communication, significantly improves the robot's processing capabilities.

In our experimentation, the direction detection system achieved an accuracy rate exceeding 90%. Additionally, decision tree, Naïve Bayes, Support Vector Machine, and k-Nearest Neighbors classifiers attained accuracy levels ranging from 90% to 99%. We also explored composite operators like exponential pessimistic, optimistic exponential, weighted average, dependent operator, and learning from observations, which, when combined with threshold considerations, reduced sensor information discrepancies, resulting in up to an 80% reduction in errors.

In conclusion, this study demonstrates the potential of integrating IoT, deep neural networks, computer vision, and human-computer interaction to create an intelligent robot capable of efficient navigation within dynamic environments. It highlights the robustness and adaptability of the proposed algorithms and systems, offering insights into the future of intelligent robotics.

![Watch the Project Gif](https://github.com/mhbadiei/Computer-Engineering-Bachelor-Thesis-AUT/blob/main/gif.gif)

Explore our project in action by watching the full version video.

[Watch the Project Video](https://github.com/mhbadiei/Computer-Engineering-Bachelor-Thesis-AUT/blob/main/video.mp4) 
