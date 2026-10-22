#include "contact_sample.hh"

#include <chrono>
#include <memory>
#include <vector>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Collision.hh>
#include <ignition/gazebo/components/ContactSensorData.hh>
#include <ignition/gazebo/components/Link.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/msgs/stringmsg.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>
#include <sdf/Element.hh>

namespace opticore
{
namespace sim = ignition::gazebo;
namespace comp = ignition::gazebo::components;

class M2ContactMonitor : public sim::System, public sim::ISystemConfigure,
                         public sim::ISystemPreUpdate, public sim::ISystemPostUpdate
{
public:
  void Configure(const sim::Entity &entity, const std::shared_ptr<const sdf::Element> &sdf,
                 sim::EntityComponentManager &, sim::EventManager &) override
  {
    world = entity;
    robot = sdf->Get<std::string>("robot", "amr1").first;
    floor = sdf->Get<std::string>("floor_model", "ground_plane").first;
    publisher = transport.Advertise<ignition::msgs::StringMsg>("/" + robot + "/m2/contacts");
  }

  void PreUpdate(const sim::UpdateInfo &, sim::EntityComponentManager &ecm) override
  {
    if (robotEntity != sim::kNullEntity && ecm.HasEntity(robotEntity) && !parts.empty())
      return;
    sampling.invalidate();
    parts.clear();
    robotEntity = ecm.EntityByComponents(comp::Model(), comp::Name(robot), comp::ParentEntity(world));
    if (robotEntity == sim::kNullEntity)
      return;
    bool wheel = false;
    bool chassis = false;
    for (auto link : ecm.ChildrenByComponents(robotEntity, comp::Link()))
    {
      for (auto collision : ecm.ChildrenByComponents(link, comp::Collision()))
      {
        // The physics system populates this component for the previous step
        // even when there are no contacts (unlike the event-only sensor topic).
        if (!ecm.Component<comp::ContactSensorData>(collision))
          ecm.CreateComponent(collision, comp::ContactSensorData());
        const auto name = sim::scopedName(collision, ecm, "::", false);
        parts.push_back({collision, name});
        wheel = wheel || isSupportPart(name);
        chassis = chassis || (!isSupportPart(name) &&
            (name.find("base_link") != std::string::npos ||
             name.find("base_footprint") != std::string::npos));
      }
    }
    // Spawn is asynchronous. Only observe after finding both collision types.
    if (!wheel || !chassis)
      parts.clear();
  }

  void PostUpdate(const sim::UpdateInfo &info, const sim::EntityComponentManager &ecm) override
  {
    if (info.paused || parts.empty())
      return;
    Contacts sample;
    for (const auto &part : parts)
    {
      const auto *data = ecm.Component<comp::ContactSensorData>(part.id);
      if (!data)
      {
        sample.complete = false;
        continue;
      }
      for (const auto &pair : data->Data().contact())
      {
        // Physics sets entity IDs even when the optional name field is empty.
        const auto first = pair.collision1().id();
        const auto second = pair.collision2().id();
        if (first != part.id && second != part.id)
        {
          sample.complete = false;
          continue;
        }
        const auto other = (first == part.id) ? second : first;
        sample.add(part.name, topLevelModel(other, ecm), robot, floor);
      }
    }
    const auto stamp = std::chrono::duration<double>(info.simTime).count();
    const auto msg = sampling.update(stamp, sample, robot, [&]() {
      const auto actor = ecm.EntityByComponents(comp::Model(),
          comp::Name("m2_mission_obstacle"), comp::ParentEntity(world));
      if (actor == sim::kNullEntity)
        return ActorObservation{};
      // A model directly parented to the world has a world-relative Pose.
      const auto *pose = ecm.Component<comp::Pose>(actor);
      if (!pose)
        return ActorObservation{std::nullopt, false};
      const auto &position = pose->Data().Pos();
      return ActorObservation{std::array<double, 2>{position.X(), position.Y()}, true};
    });
    if (msg)
    {
      ignition::msgs::StringMsg report;
      report.set_data(*msg);
      publisher.Publish(report);
    }
  }

private:
  struct Part { sim::Entity id; std::string name; };

  // Walk through links or nested models to the independently spawned entity.
  std::string topLevelModel(sim::Entity entity, const sim::EntityComponentManager &ecm) const
  {
    while (entity != sim::kNullEntity && entity != world)
    {
      auto parent = ecm.ParentEntity(entity);
      if (parent == world && ecm.Component<comp::Model>(entity))
      {
        const auto *name = ecm.Component<comp::Name>(entity);
        return name ? name->Data() : "";
      }
      entity = parent;
    }
    return "";
  }

  sim::Entity world = sim::kNullEntity;
  sim::Entity robotEntity = sim::kNullEntity;
  std::string robot;
  std::string floor;
  ignition::transport::Node transport;
  ignition::transport::Node::Publisher publisher;
  std::vector<Part> parts;
  Sampling sampling;
};
}

IGNITION_ADD_PLUGIN(opticore::M2ContactMonitor, ignition::gazebo::System,
    opticore::M2ContactMonitor::ISystemConfigure,
    opticore::M2ContactMonitor::ISystemPreUpdate,
    opticore::M2ContactMonitor::ISystemPostUpdate)
IGNITION_ADD_PLUGIN_ALIAS(opticore::M2ContactMonitor, "opticore::M2ContactMonitor")
