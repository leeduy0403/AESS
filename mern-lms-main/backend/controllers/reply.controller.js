import { errorHandler } from "../utils/error.js";
import Question from "../models/question.model.js";
import Reply from "../models/reply.model.js";

export const getReplies = async (req, res, next) => {
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.sort === "desc" ? -1 : 1;
    const replies = await Question.find({ questionId: req.params.questionId })
      .sort({ createdAt: sortDirection })
      .skip(startIndex)
      .limit(limit);
    res.status(200).json(replies);
  } catch (error) {
    next(error);
  }
};

export const createReply = async (req, res, next) => {
  try {
    const { content, userId, questionId } = req.body;
    const question = await Question.findById(questionId);
    if (!question) {
      return next(errorHandler(404, "Question not found"));
    }
    const newReply = new Reply({
      content,
      userId,
      questionId,
    });
    await newReply.save();
    question.replies.push(newReply._id);
    await question.save();
    res.status(200).json(newReply);
  } catch (error) {
    next(error);
  }
};

export const editReply = async (req, res, next) => {
  try {
    const reply = await Reply.findById(req.params.replyId);
    if (!reply) {
      return next(errorHandler(404, "Reply not found"));
    }
    const editedReply = await Reply.findByIdAndUpdate(
      req.params.replyId,
      {
        content: req.body.content,
      },
      { new: true }
    );
    res.status(200).json(editedReply);
  } catch (error) {
    next(error);
  }
};

export const deleteReply = async (req, res, next) => {
  try {
    const question = await Question.findById(req.params.questionId);
    const reply = await Reply.findById(req.params.replyId);
    if (!question) {
      return next(errorHandler(404, "question not found"));
    }
    if (!reply) {
      return next(errorHandler(404, "reply not found"));
    }
    question.replies = question.replies.filter(
      (item) => item.toString() !== req.params.replyId
    );
    await question.save();
    await Reply.findByIdAndDelete(req.params.replyId);
    res.status(200).json("Reply has been deleted");
  } catch (error) {
    next(error);
  }
};
