// Preserve the middleware writer identity for each actual goal sample on Humble.
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  using Goal = geometry_msgs::msg::PoseStamped;
  using Observation = std_msgs::msg::String;
  auto node = std::make_shared<rclcpp::Node>("m5_goal_witness");
  auto observed = node->create_publisher<Observation>("/m5/observed_goals", 32);
  std::vector<rclcpp::Subscription<Goal>::SharedPtr> subscriptions;
  for (const std::string robot : {"amr1", "amr2"}) {
    auto subscription = node->create_subscription<Goal>(
        "/" + robot + "/goal_pose", 32,
        [robot, observed](Goal::ConstSharedPtr goal, const rclcpp::MessageInfo& info) {
          const auto& gid = info.get_rmw_message_info().publisher_gid;
          std::ostringstream row;
          row << "{\"robot\":\"" << robot << "\",\"writer\":\"" << std::hex << std::setfill('0');
          for (const auto octet : gid.data) {
            row << std::setw(2) << static_cast<unsigned>(octet);
          }
          const auto& p = goal->pose.position;
          const auto& q = goal->pose.orientation;
          row << std::dec << std::setprecision(17)
              << "\",\"is_map\":" << (goal->header.frame_id == "map" ? "true" : "false")
              << ",\"sec\":" << goal->header.stamp.sec << ",\"nsec\":" << goal->header.stamp.nanosec
              << ",\"xy\":[" << p.x << "," << p.y << "],\"orientation\":[" << q.x << "," << q.y
              << "," << q.z << "," << q.w << "]}";
          Observation msg;
          msg.data = row.str();
          observed->publish(msg);
        });
    subscriptions.push_back(subscription);
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
