#include "contact_sample.hh"

#include <iostream>
#include <stdexcept>

namespace
{
void check(bool predicate, const char *msg)
{
  if (!predicate) throw std::runtime_error(msg);
}
bool contains(const std::optional<std::string> &sample, const std::string &part)
{
  return sample && sample->find(part) != std::string::npos;
}
}

int main()
{
  opticore::Sampling sampler;
  opticore::Contacts clear;
  const std::string robot = "amr1";
  const std::string floor = "ground_plane";
  check(!sampler.update(2.0, clear, robot), "absent physics must be unknown");
  clear.add("left_wheel_link::collision", floor, robot, floor);
  clear.add("base_link::caster_collision", robot, robot, floor);
  auto start = sampler.update(2.1, clear, robot);
  check(contains(start, "\"collision\":false") && contains(start, "\"pairs\":[]"), "ground contact");
  check(contains(start, "\"actor_world_xy\":null") && contains(start, "\"coverage_epoch\":0"),
        "absence confirmed at episode start");
  check(!sampler.update(2.11, clear, robot), "no heartbeat flood");
  check(contains(sampler.update(2.16, clear, robot), "\"collision\":false"), "clear heartbeat");

  auto rack = clear;
  rack.add("base_link", "rack_A1", robot, floor);
  check(contains(sampler.update(2.164, rack, robot), "[\"amr1\",\"rack_A1\"]"), "static rack hit");
  auto normal = sampler.update(2.168, clear, robot);
  check(contains(normal, "\"collision\":false") && contains(normal, "\"collision_events\":1"),
        "four millisecond edge survives even between heartbeats");
  auto moving = clear;
  moving.add("right_wheel_link", "m2_mission_obstacle", robot, floor);
  check(contains(sampler.update(2.172, moving, robot), "m2_mission_obstacle"), "moving actor hits wheel");
  opticore::Contacts tipped;
  tipped.add("base_link::base_link_collision", floor, robot, floor);
  check(contains(sampler.update(2.23, tipped, robot), "\"collision\":true"), "chassis struck floor");
  opticore::Contacts unsupported;
  check(!sampler.update(2.50, unsupported, robot), "stale physics evidence expired");

  auto missing = clear;
  missing.add("base_link", "", robot, floor);
  check(!sampler.update(2.51, missing, robot), "unresolved partner must be unknown");
  check(!sampler.update(0.01, unsupported, robot), "rewind loses previous episode support");
  check(contains(sampler.update(0.02, clear, robot), "\"collision_events\":0"), "rewind clears counter");
  clear.support = true;
  opticore::Sampling gap;
  gap.update(10.0, clear, robot);
  check(!gap.update(10.004, missing, robot), "incomplete physics step suppressed");
  check(!gap.update(10.008, missing, robot), "same incomplete period");
  auto recovered = gap.update(10.012, clear, robot, [] {
    return opticore::ActorObservation{std::array<double, 2>{8.0, -2.5}, true};
  });
  check(contains(recovered, "\"coverage_epoch\":1"), "sub-heartbeat unknown period is sticky");
  check(contains(recovered, "\"actor_world_xy\":[8,-2.5]") && contains(recovered, "\"actor_pose_valid\":true"),
        "world actor observation");
  gap.invalidate();
  auto absent = gap.update(10.016, clear, robot);
  check(contains(absent, "\"coverage_epoch\":2") && contains(absent, "\"actor_world_xy\":null") &&
        contains(absent, "\"actor_pose_valid\":true"), "robot replacement and actor removal");
  auto invalidPose = gap.update(10.12, clear, robot, [] {
    return opticore::ActorObservation{std::nullopt, false};
  });
  check(contains(invalidPose, "\"actor_pose_valid\":false") && contains(invalidPose, "\"actor_world_xy\":null"),
        "missing actor pose is distinct from confirmed absence");
  check(opticore::quoteJson("a\"b") == "\"a\\\"b\"", "JSON escapes names");
  std::cout << "contact classification, support freshness, short collision and rewind passed\n";
}
