import { errorHandler } from "../utils/error.js";
import Group from "../models/group.model.js";
import Assignment from "../models/assignment.model.js";
import Class from "../models/class.model.js";
import User from "../models/user.model.js";

export const getGroupsOfAssignment = async (req, res, next) => {
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    const groups = await Group.find({
      _id: { $in: assignment.groups },
      ...(req.query.groupId && { _id: req.query.groupId }),
    });
    res.status(200).json(groups);
  } catch (error) {
    next(error);
  }
};

export const getGroupOfAssignmentOfUser = async (req, res, next) => {
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    const group = await Group.findOne({
      _id: { $in: assignment.groups },
      members: req.params.userId,
    }).populate({
      path: "members",
      model: User,
    });
    if (!group) {
      return res
        .status(404)
        .json({ message: "User is not in any group for this assignment." });
    }
    res.status(200).json(group);
  } catch (error) {
    next(error);
  }
};

export const createGroup = async (req, res, next) => {
  const assignment = await Assignment.findById(req.params.assignmentId);
  if (!req.body.name) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newGroup = new Group({ ...req.body });
  try {
    assignment.groups.push(newGroup._id);
    await assignment.save();
    const savedGroup = await newGroup.save();
    res.status(200).json(savedGroup);
  } catch (error) {
    next(error);
  }
};

export const joinGroup = async (req, res, next) => {
  const assignment = await Assignment.findById(req.params.assignmentId);
  const user = await User.findById(req.body.userId);
  try {
    if (user.isEducator) {
      return next(errorHandler(403, "Educators cannot join groups!"));
    }
    const group = await Group.findById(req.body.groupId);
    if (!group) {
      return next(errorHandler(404, "Group not found!"));
    }
    if (group.members.length >= assignment.maxMemberGroup) {
      return next(errorHandler(403, "Group is full!"));
    }
    if (group.members.includes(req.body.userId)) {
      return next(errorHandler(403, "You are already in this group!"));
    }
    const currentGroup = await Group.findOne({
      _id: { $in: assignment.groups },
      members: req.body.userId,
    });
    if (currentGroup) {
      currentGroup.members = currentGroup.members.filter(
        (member) => member.toString() !== req.body.userId
      );
      await currentGroup.save();
    }
    group.members.push(req.body.userId);
    await group.save();
    res.status(200).json(group);
  } catch (error) {
    next(error);
  }
};

export const leaveGroup = async (req, res, next) => {
  try {
    const group = await Group.findById(req.body.groupId);
    if (!group) {
      return next(errorHandler(404, "Group not found!"));
    }
    if (!group.members.includes(req.body.userId)) {
      return next(errorHandler(403, "You are not in this group!"));
    }
    group.members = group.members.filter((item) => item !== req.body.userId);
    await group.save();
    res.status(200).json(group);
  } catch (error) {
    next(error);
  }
};

export const removeMemberOfGroup = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(errorHandler(403, "You are not an educator of this class!"));
  }
  try {
    const group = await Group.findById(req.body.groupId);
    if (!group) {
      return next(errorHandler(404, "Group not found!"));
    }
    if (!group.members.includes(req.body.userId)) {
      return next(errorHandler(403, "You are not in this group!"));
    }
    group.members = group.members.filter((item) => item !== req.body.userId);
    await group.save();
    res.status(200).json(group);
  } catch (error) {
    next(error);
  }
};

export const deleteGroup = async (req, res, next) => {
  const assignment = await Assignment.findById(req.params.assignmentId);
  if (!assignment) {
    return next(errorHandler(404, "Assignment not found!"));
  }
  try {
    assignment.groups = assignment.groups.filter(
      (item) => item.toString() !== req.params.groupId
    );
    await assignment.save();
    await Group.findByIdAndDelete(req.params.groupId);
    res.status(200).json("Group has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateGroup = async (req, res, next) => {
  try {
    const updatedGroup = await Group.findByIdAndUpdate(
      req.params.groupId,
      {
        $set: {
          name: req.body.name,
          maxMember: req.body.maxMember,
          members: req.body.members,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedGroup);
  } catch (error) {
    next(error);
  }
};
