import { errorHandler } from "../utils/error.js";
import Topic from "../models/topic.model.js";
import Question from "../models/question.model.js";
import Submission from "../models/submission.model.js";

export const getQuestions = async (req, res, next) => {
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.sort === "desc" ? -1 : 1;
    const topic = await Topic.findById(req.params.topicId).populate({
      path: "userId",
      model: "User",
      select: "name profilePicture",
    });
    if (!topic) {
      return next(errorHandler(404, "Topic not found"));
    }
    const questions = await Question.find({ topicId: req.params.topicId })
      .sort({ createdAt: sortDirection })
      .skip(startIndex)
      .limit(limit)
      .populate({
        path: "replies",
        model: "Reply",
        populate: {
          path: "userId",
          model: "User",
          select: "name profilePicture",
        },
      })
      .populate({
        path: "userId",
        model: "User",
        select: "name profilePicture",
      });
    res.status(200).json({ questions, topicInfo: topic });
  } catch (error) {
    next(error);
  }
};

export const createQuestion = async (req, res, next) => {
  try {
    const { content, userId, topicId } = req.body;
    const topic = await Topic.findById(topicId);
    if (!topic) {
      return next(errorHandler(404, "Topic not found"));
    }
    if (!content) {
      return next(errorHandler(400, "Content is required"));
    }
    const newQuestion = new Question({
      content,
      userId,
      topicId,
    });
    await newQuestion.save();
    topic.questions.push(newQuestion._id);
    await topic.save();
    res.status(200).json(newQuestion);
  } catch (error) {
    next(error);
  }
};

export const createQuestionFromSubmission = async (req, res, next) => {
  try {
    const updates = Object.entries(req.body.requestData);
    if (updates.length === 0) {
      res.status(200).json({ message: "Nothing to update" });
    }
    await Promise.all(
      updates.map(async ([submissionId, content]) => {
        const submission = await Submission.findById(submissionId);
        if (!submission) {
          return next(errorHandler(404, "Submission not found"));
        }
        const topicId = submission.reviewRequest;
        const topic = await Topic.findById(topicId);
        if (!topic) {
          return next(errorHandler(404, "Topic not found"));
        }
        if (!content) {
          return next(errorHandler(400, "Content is required"));
        }
        const newQuestion = new Question({
          content,
          userId: req.body.userId,
          topicId,
        });
        await newQuestion.save();
        topic.questions.push(newQuestion._id);
        await topic.save();
      })
    );
    res.status(200).json({ message: "Sending request(s) successfully" });
  } catch (error) {
    next(error);
  }
};

export const editQuestion = async (req, res, next) => {
  try {
    const question = await Question.findById(req.params.questionId);
    if (!question) {
      return next(errorHandler(404, "Question not found"));
    }
    const editedQuestion = await Question.findByIdAndUpdate(
      req.params.questionId,
      {
        content: req.body.content,
      },
      { new: true }
    );
    res.status(200).json(editedQuestion);
  } catch (error) {
    next(error);
  }
};

export const deleteQuestion = async (req, res, next) => {
  try {
    const topic = await Topic.findById(req.params.topicId);
    const question = await Question.findById(req.params.questionId);
    if (!topic) {
      return next(errorHandler(404, "Topic not found"));
    }
    if (!question) {
      return next(errorHandler(404, "Question not found"));
    }
    topic.questions = topic.questions.filter(
      (item) => item.toString() !== req.params.questionId
    );
    await topic.save();
    await Question.findByIdAndDelete(req.params.questionId);
    res.status(200).json("Question has been deleted");
  } catch (error) {
    next(error);
  }
};
