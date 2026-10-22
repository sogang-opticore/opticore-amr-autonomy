#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>

namespace opticore
{
inline bool isSupportPart(const std::string &name)
{
  return name.find("wheel") != std::string::npos ||
         name.find("caster") != std::string::npos;
}

// One physics step, including the complete set of collision entities on the AMR.
struct Contacts
{
  bool complete = true;
  bool support = false;
  std::set<std::string> otherModels;

  void add(const std::string &part, const std::string &other,
           const std::string &robot, const std::string &floor)
  {
    if (other.empty())
      complete = false;
    else if (other == floor && isSupportPart(part))
      support = true;
    else if (other != robot)
      otherModels.insert(other);
  }
};

inline std::string quoteJson(const std::string &s)
{
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s)
  {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<unsigned>(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}

struct ActorObservation
{
  // No actor is a valid observation. A present actor with missing pose is not.
  std::optional<std::array<double, 2>> position;
  bool valid = true;
};

class Sampling
{
  // No negative samples until wheel/floor contact demonstrates a working
  // physics stream. Brief wheel bounce is tolerated, loss of support expires.
  double lastSupport = -1.0;
  double lastPublication = -1.0;
  bool previousCollision = false;
  std::uint64_t collisionEvents = 0;
  double lastStep = -1.0;
  std::uint64_t coverageEpoch = 0;
  bool covered = false;

public:
  void invalidate()
  {
    lastSupport = -1.0;
    if (covered)
    {
      // This epoch survives lost packets and even a one-physics-step gap.
      ++coverageEpoch;
      previousCollision = false;
      lastPublication = -1.0;
      covered = false;
    }
  }

  std::optional<std::string> update(double time, const Contacts &sample,
                                    const std::string &robot,
                                    const std::function<ActorObservation()> &observeActor =
                                        [] { return ActorObservation{}; })
  {
    if (!std::isfinite(time) || time < 0.0)
      return std::nullopt;
    if (time < lastStep)
      *this = Sampling();
    lastStep = time;
    if (sample.support)
      lastSupport = time;
    const bool collision = !sample.otherModels.empty();
    const bool healthy = sample.complete && (collision ||
        (lastSupport >= 0.0 && time - lastSupport <= 0.15));
    if (!healthy)
      invalidate();
    covered = healthy;
    if (!healthy && !collision)
      return std::nullopt;
    const bool edge = collision != previousCollision;
    if (collision && edge)
      ++collisionEvents;
    previousCollision = collision;
    if (!edge && lastPublication >= 0 && time - lastPublication < 0.05)
      return std::nullopt;
    lastPublication = time;
    std::ostringstream out;
    const ActorObservation actor = observeActor();
    const bool poseValid = actor.valid && (!actor.position ||
        (std::isfinite((*actor.position)[0]) && std::isfinite((*actor.position)[1])));
    out << std::setprecision(17)
        << "{\"source\":\"gazebo_contacts\",\"stamp\":" << time
        << ",\"collision\":" << (collision ? "true" : "false")
        << ",\"collision_events\":" << collisionEvents
        << ",\"coverage_epoch\":" << coverageEpoch
        << ",\"actor_pose_valid\":" << (poseValid ? "true" : "false")
        << ",\"actor_world_xy\":";
    if (poseValid && actor.position)
      out << '[' << (*actor.position)[0] << ',' << (*actor.position)[1] << ']';
    else
      out << "null";
    out << ",\"pairs\":[";
    bool first = true;
    for (const auto &other : sample.otherModels)
    {
      if (!first)
        out << ',';
      out << '[' << quoteJson(robot) << ',' << quoteJson(other) << ']';
      first = false;
    }
    out << "]}";
    return out.str();
  }
};
}  // namespace opticore
