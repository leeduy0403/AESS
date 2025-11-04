import { errorHandler } from "../utils/error.js";
import Class from "../models/class.model.js";
import Forum from "../models/forum.model.js";
import Topic from "../models/topic.model.js";
import Question from "../models/question.model.js";

export const getForums = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    if (!classItem) {
      return next(errorHandler(404, "Class not found"));
    }
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.sort === "desc" ? -1 : 1;
    // const forums = await Forum.find({ classId: req.params.classId })
    const forums = await Forum.find({ _id: classItem.forums })
      .sort({ createdAt: sortDirection })
      .skip(startIndex)
      .limit(limit)
      .lean();
    for (let forum of forums) {
      const topics = await Topic.find({ _id: { $in: forum.topics } })
        .populate({
          path: "userId",
          model: "User",
          select: "name profilePicture",
        })
        .lean();
      for (let topic of topics) {
        const questions = await Question.find({
          _id: { $in: topic.questions },
        }).lean();
        let totalReplies = 0;
        let latestUpdate = new Date(topic.updatedAt);
        for (const question of questions) {
          totalReplies += question.replies?.length || 0;
          if (new Date(question.updatedAt) > latestUpdate) {
            latestUpdate = new Date(question.updatedAt);
          }
        }
        topic.totalReplies = totalReplies;
        topic.lastUpdated = latestUpdate;
      }
      forum.topics = topics;
    }
    res.status(200).json(forums);
  } catch (error) {
    next(error);
  }
};

export const createForum = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.body.classId);
    if (!classItem) {
      return next(errorHandler(404, "Class not found"));
    }
    const newForum = new Forum({
      title: req.body.title,
      classId: req.body.classId,
    });
    await newForum.save();
    classItem.forums.push(newForum._id);
    await classItem.save();
    res.status(200).json(newForum);
  } catch (error) {
    next(error);
  }
};

export const editForum = async (req, res, next) => {
  try {
    const forum = await Forum.findById(req.params.forumId);
    if (!forum) {
      return next(errorHandler(404, "forum not found"));
    }
    const editedForum = await Forum.findByIdAndUpdate(
      req.params.forumId,
      {
        $set: {
          title: req.body.title,
        },
      },
      { new: true }
    );
    res.status(200).json(editedForum);
  } catch (error) {
    next(error);
  }
};

export const deleteForum = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const forum = await Forum.findById(req.params.forumId);
    if (!classItem) {
      return next(errorHandler(404, "Class not found"));
    }
    if (!forum) {
      return next(errorHandler(404, "Forum not found"));
    }
    classItem.forums = classItem.forums.filter(
      (item) => item.toString() !== req.params.forumId
    );
    await classItem.save();
    await Forum.findByIdAndDelete(req.params.forumId);
    res.status(200).json("Forum has been deleted");
  } catch (error) {
    next(error);
  }
};
