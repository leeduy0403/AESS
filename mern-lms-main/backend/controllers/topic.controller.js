import { errorHandler } from "../utils/error.js";
import Forum from "../models/forum.model.js";
import Topic from "../models/topic.model.js";

export const getTopics = async (req, res, next) => {
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.sort === "desc" ? -1 : 1;
    const topics = await Topic.find({ forumId: req.params.forumId })
      .sort({ createdAt: sortDirection })
      .skip(startIndex)
      .limit(limit);
    res.status(200).json(topics);
  } catch (error) {
    next(error);
  }
};

export const createTopic = async (req, res, next) => {
  try {
    const { content, userId, forumId } = req.body;
    const forum = await Forum.findById(forumId);
    if (!forum) {
      return next(errorHandler(404, "Forum not found"));
    }
    if (!content) {
      return next(errorHandler(400, "Content is required"));
    }
    const newTopic = new Topic({
      content,
      userId,
      forumId,
    });
    await newTopic.save();
    forum.topics.push(newTopic._id);
    await forum.save();
    res.status(200).json(newTopic);
  } catch (error) {
    next(error);
  }
};

export const editTopic = async (req, res, next) => {
  try {
    const topic = await Topic.findById(req.params.topicId);
    if (!topic) {
      return next(errorHandler(404, "Topic not found"));
    }
    const editedTopic = await Topic.findByIdAndUpdate(
      req.params.topicId,
      {
        $set: {
          content: req.body.content,
        },
      },
      { new: true }
    );
    res.status(200).json(editedTopic);
  } catch (error) {
    next(error);
  }
};

export const deleteTopic = async (req, res, next) => {
  try {
    const forum = await Forum.findById(req.params.forumId);
    const topic = await Topic.findById(req.params.topicId);
    if (!forum) {
      return next(errorHandler(404, "Forum not found"));
    }
    if (!topic) {
      return next(errorHandler(404, "Topic not found"));
    }
    forum.topics = forum.topics.filter(
      (item) => item.toString() !== req.params.topicId
    );
    await forum.save();
    await Topic.findByIdAndDelete(req.params.topicId);
    res.status(200).json("Topic has been deleted");
  } catch (error) {
    next(error);
  }
};
