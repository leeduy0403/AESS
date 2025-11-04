import { errorHandler } from "../utils/error.js";
import Class from "../models/class.model.js";
import Material from "../models/material.model.js";
import Group from "../models/group.model.js";
import Assignment from "../models/assignment.model.js";
import Topic from "../models/topic.model.js";
import Submission from "../models/submission.model.js";

export const getSubmissions = async (req, res, next) => {
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    const submissions = await Submission.find({
      _id: { $in: assignment.submissions },
      ...(req.query.submissionId && { _id: req.query.submissionId }),
    });
    res.status(200).json(submissions);
  } catch (error) {
    next(error);
  }
};

export const getSubmissionsOfAssignmentOfUser = async (req, res, next) => {
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    if (assignment.type === "Individual") {
      const submissions = await Submission.find({
        _id: { $in: assignment.submissions },
        uploadBy: req.params.userId,
      });
      res.status(200).json(submissions);
    } else if (assignment.type === "Group") {
      const group = await Group.findOne({
        _id: { $in: assignment.groups },
        members: req.params.userId,
      });
      if (!group) {
        return res.status(404).json({ message: "Group not found" });
      }
      const submissions = await Submission.find({
        _id: { $in: assignment.submissions },
        groupId: group._id,
      });
      res.status(200).json(submissions);
    } else {
      return res.status(404).json({ message: "Invalid assignment type" });
    }
  } catch (error) {
    next(error);
  }
};

export const createSubmission = async (req, res, next) => {
  if (
    !req.body.submissionUrls ||
    !req.body.submissionFormats ||
    !req.body.nameFiles ||
    !req.body.uploadBy
  ) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const assignment = await Assignment.findById(req.params.assignmentId);
  if (!assignment) {
    return next(errorHandler(404, "Assignment not found"));
  }
  if (assignment.startDate > new Date()) {
    return next(errorHandler(403, "Assignment has not started yet!"));
  }
  if (assignment.endDate < new Date()) {
    return next(errorHandler(403, "Assignment has ended!"));
  }
  const submissions = await Submission.find({
    _id: { $in: assignment.submissions },
  });
  if (assignment.type === "Individual") {
    if (
      submissions.filter(
        (submission) => submission.uploadBy === req.body.uploadBy
      ).length >= assignment.maxAttempt
    ) {
      return next(
        errorHandler(403, "You have reached the maximum number of attempts!")
      );
    }
  }
  if (assignment.type === "Group") {
    if (
      submissions.filter(
        (submission) => submission.groupId === req.body.groupId
      ).length >= assignment.maxAttempt
    ) {
      return next(
        errorHandler(
          403,
          "You have reached the maximum number of attempts for this group!"
        )
      );
    }
    const group = await Group.findById(req.body.groupId);
    if (!group) {
      return next(errorHandler(404, "Group not found!"));
    }
    if (!group.members.includes(req.body.uploadBy)) {
      return next(
        errorHandler(403, "You are not allowed to submit for this group!")
      );
    }
  }
  if (assignment.submissionFormats.length > 0) {
    if (
      !req.body.submissionFormats.every((item) =>
        assignment.submissionFormats.includes(item)
      )
    ) {
      return next(
        errorHandler(
          403,
          "This submission format is not allowed for this assignment!"
        )
      );
    }
  }
  if (assignment.totalFileSize < req.body.totalFileSize) {	//! No field named totalFileSize in req.body
    console.log("Exceeds file size limit: ", req.body.totalFileSize); //? debug
    console.log("Max file size limit: ", assignment.totalFileSize); //? debug
    return next(
      errorHandler(
        403,
        "Your submission file size exceeds the maximum allowed size!"
      )
    );
  }
  if (req.body.submissionUrls.length > assignment.maxNumberOfFile) {
    return next(
      errorHandler(
        403,
        "Your submission file count exceeds the maximum allowed number!"
      )
    );
  }
  try {
    const newSubmission = new Submission({ ...req.body });
    const newReviewRequest = new Topic({
      content: assignment.title,
    });
    await newReviewRequest.save();
    newSubmission.reviewRequest = newReviewRequest._id;
    await newSubmission.save();
    assignment.submissions.push(newSubmission._id);
    await assignment.save();
    res.status(200).json(newSubmission);
  } catch (error) {
    next(error);
  }
};

export const deleteSubmission = async (req, res, next) => {
  try {
    classItem.assignments = classItem.assignments.filter(
      (item) => item !== req.params.assignmentId
    );
    await classItem.save();
    const assignment = await Assignment.findById(req.params.assignmentId);
    const materials = await Material.find({
      _id: { $in: assignment.materials },
    });
    await Promise.all(
      materials.map(async (material) => {
        await Material.findByIdAndDelete(material._id);
      })
    );
    await Assignment.findByIdAndDelete(req.params.assignmentId);
    res.status(200).json("Assignment has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateSubmission = async (req, res, next) => {
  try {
    const submission = await Submission.findById(req.params.submissionId);
    if (!submission) {
      return next(errorHandler(404, "Submission not found"));
    }
    const updatedSubmission = await Submission.findByIdAndUpdate(
      req.params.submissionId,
      {
        $set: {
          description: req.body.description,
          submissionUrls: req.body.submissionUrls,
          nameFiles: req.body.nameFiles,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedSubmission);
  } catch (error) {
    next(error);
  }
};

export const updateMemberScores = async (req, res, next) => {
  try {
    const updates = Object.entries(req.body.formData);
    if (updates.length === 0) {
      res.status(200).json({ message: "Nothing to update" });
    }
    await Promise.all(
      updates.map(async ([submissionId, score]) => {
        const submission = await Submission.findById(submissionId).lean();
        if (!submission) return;
        if (submission.groupId) {
          await Submission.findByIdAndUpdate(submissionId, {
            $set: { individualScores: score },
          });
        } else {
          await Submission.findByIdAndUpdate(submissionId, {
            $set: { individualScores: score },
          });
        }
      })
    );
    res.status(200).json({ message: "Scores updated successfully" });
  } catch (error) {
    next(error);
  }
};
