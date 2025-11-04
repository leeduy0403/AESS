import { errorHandler } from "../utils/error.js";
import Class from "../models/class.model.js";
import Material from "../models/material.model.js";
import Assignment from "../models/assignment.model.js";
import Forum from "../models/forum.model.js";
import Submission from "../models/submission.model.js";
import Group from "../models/group.model.js";
import User from "../models/user.model.js";
import Key from "../models/key.model.js";
import schedule from "node-schedule";
import mongoose from "mongoose";

export const getAssignments = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const assignments = await Assignment.find({
      _id: { $in: classItem.assignments },
      ...(req.query.assignmentId && { _id: req.query.assignmentId }),
    }).populate({
      path: "materials",
      model: "Material",
    });
    const assignmentsWithSubmissions = await Promise.all(
      assignments.map(async (assignment) => {
        const lastSubmission = await Submission.findById(
          assignment.submissions[assignment.submissions.length - 1]
        ).lean();
        const groups = await Group.find({ _id: { $in: assignment.groups } })
          .populate({
            path: "members",
            model: User,
          })
          .lean();
        return {
          ...assignment.toObject(),
          lastSubmission,
          groups,
        };
      })
    );
    const now = new Date();
    const ongoingAssignments = await Assignment.find({
      endDate: { $gte: now },
    }).sort({ endDate: 1 });
    res
      .status(200)
      .json({ assignments: assignmentsWithSubmissions, ongoingAssignments });
  } catch (error) {
    next(error);
  }
};

export const getOngoingAssignments = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const now = new Date();
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 5;
    const ongoingAssignments = await Assignment.find({
      _id: { $in: classItem.assignments },
      endDate: { $gte: now },
    })
      .skip(startIndex)
      .limit(limit)
      .sort({ endDate: 1 });
    res.status(200).json(ongoingAssignments);
  } catch (error) {
    next(error);
  }
};

export const createAssignment = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to create an assignment for this class!"
      )
    );
  }
  if (!req.body.title) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  try {
    const newAssignment = new Assignment({
      ...req.body,
      classId: req.params.classId,
    });
    await newAssignment.save();
    const newForum = new Forum({
      title: newAssignment.title,
      classId: req.params.classId,
    });
    await newForum.save();
    classItem.assignments.push(newAssignment._id);
    classItem.forums.push(newForum._id);
    await classItem.save();
    res.status(200).json(newAssignment);
  } catch (error) {
    next(error);
  }
};

export const deleteAssignment = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to delete an assignment for this class!"
      )
    );
  }
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

export const updateAssignment = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to update an assignment for this class!"
      )
    );
  }
  try {
    const updatedAssignment = await Assignment.findByIdAndUpdate(
      req.params.assignmentId,
      {
        $set: {
          title: req.body.title,
          description: req.body.description,
          startDate: req.body.startDate,
          endDate: req.body.endDate,
          triggerDate: req.body.triggerDate,
          type: req.body.type,
          status: req.body.status,
          isHidden: req.body.isHidden,
          isScorePublish: req.body.isScorePublish,
          publishDate: req.body.publishDate,
          allowModify: req.body.allowModify,
          autoEvaluate: req.body.autoEvaluate,
          submissionFormat: req.body.submissionFormat,
          maxNumberOfFile: req.body.maxNumberOfFile,
          maxAttempt: req.body.maxAttempt,
          totalFileSize: req.body.totalFileSize,
          gradingStatus: req.body.gradingStatus,
          maxMemberGroup: req.body.maxMemberGroup,
          startDateGroup: req.body.startDateGroup,
          endDateGroup: req.body.endDateGroup,
          groups: req.body.groups,
          materials: req.body.materials,
          descriptions: req.body.descriptions,
          descriptionNameFiles: req.body.descriptionNameFiles,
          rubrics: req.body.rubrics,
          rubricNameFiles: req.body.rubricNameFiles,
          submissions: req.body.submissions,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedAssignment);
  } catch (error) {
    next(error);
  }
};

export const getLastSubmissionOfAssignmentsOfUser = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const assignments = await Assignment.find({
      _id: { $in: classItem.assignments },
      ...(req.query.assignmentId && { _id: req.query.assignmentId }),
    });
    const response = [];
    for (const assignment of assignments) {
      let lastSubmission;
      let userGroup;
      if (assignment.type === "Individual") {
        lastSubmission = await Submission.findOne({
          _id: { $in: assignment.submissions },
          uploadBy: req.params.userId,
        })
          .populate({
            path: "userRequests",
            model: "User",
          })
          .sort({
            createdAt: -1,
          });
      } else if (assignment.type === "Group") {
        userGroup = await Group.findOne({
          _id: { $in: assignment.groups },
          members: req.params.userId,
        });
        if (userGroup) {
          lastSubmission = await Submission.findOne({
            _id: { $in: assignment.submissions },
            groupId: userGroup._id,
          })
            .populate({
              path: "userRequests",
              model: "User",
            })
            .sort({
              createdAt: -1,
            });
        }
      } else {
        return res.status(400).json({ message: "Invalid assignment type" });
      }
      response.push({
        assignment,
        lastSubmission,
        userGroup,
      });
    }
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

export const getLastSubmissionOfAssignmentsOfUserV2 = async (
  req,
  res,
  next
) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const assignments = await Assignment.find({
      _id: { $in: classItem.assignments },
      ...(req.query.assignmentId && { _id: req.query.assignmentId }),
    });
    const response = [];
    for (const assignment of assignments) {
      let lastSubmissionsResponse = [];
      if (assignment.type === "Individual") {
        let lastSubmission;
        lastSubmission = await Submission.findOne({
          _id: { $in: assignment.submissions },
          uploadBy: req.params.userId,
        }).sort({ createdAt: -1 });
        if (lastSubmission) {
          lastSubmissionsResponse.push({
            submission_id: lastSubmission._id,
            submission_urls: lastSubmission.submissionUrls,
            overallScore: lastSubmission.overallScore,
          });
        }
      } else if (assignment.type === "Group") {
        const groups = await Group.find({
          _id: { $in: assignment.groups },
          members: req.params.userId,
        });
        await Promise.all(
          groups.map(async (group) => {
            let lastSubmission;
            const groupInfo = await Group.findById(group._id);
            lastSubmission = await Submission.findOne({
              _id: { $in: assignment.submissions },
              groupId: groupInfo._id,
            })
              .sort({ createdAt: -1 })
              .lean();
            if (lastSubmission) {
              lastSubmissionsResponse.push({
                submission_id: lastSubmission._id,
                submission_urls: lastSubmission.submissionUrls,
                overallScore: lastSubmission.overallScore,
              });
            }
          })
        );
      } else {
        return res.status(400).json({ message: "Invalid assignment type" });
      }
      response.push({
        assignment,
        descriptions: assignment.descriptions,
        rubrics: assignment.rubrics,
        submissions: lastSubmissionsResponse,
      });
    }
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

export const getLastSubmissionOfAssignmentsOfUsers = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const assignments = await Assignment.find({
      _id: { $in: classItem.assignments },
      ...(req.query.assignmentId && { _id: req.query.assignmentId }),
    });
    const response = [];
    for (const assignment of assignments) {
      let lastSubmissionsResponse = [];
      if (assignment.type === "Individual") {
        const students = classItem.students;
        await Promise.all(
          students.map(async (studentId) => {
            let lastSubmission;
            lastSubmission = await Submission.findOne({
              _id: { $in: assignment.submissions },
              uploadBy: studentId,
            }).sort({ createdAt: -1 });
            if (lastSubmission) {
              lastSubmissionsResponse.push({
                submission_id: lastSubmission._id,
                submission_urls: lastSubmission.submissionUrls,
                overallScore: lastSubmission.overallScore,
              });
            }
          })
        );
      } else if (assignment.type === "Group") {
        const groups = await Group.find({
          _id: { $in: assignment.groups },
        });
        await Promise.all(
          groups.map(async (group) => {
            let lastSubmission;
            const groupInfo = await Group.findById(group._id);
            lastSubmission = await Submission.findOne({
              _id: { $in: assignment.submissions },
              groupId: groupInfo._id,
            })
              .sort({ createdAt: -1 })
              .lean();
            if (lastSubmission) {
              lastSubmissionsResponse.push({
                submission_id: lastSubmission._id,
                submission_urls: lastSubmission.submissionUrls,
                overallScore: lastSubmission.overallScore,
              });
            }
          })
        );
      } else {
        return res.status(400).json({ message: "Invalid assignment type" });
      }
      let averageScore =
        lastSubmissionsResponse.reduce(
          (sum, submission) => sum + (submission?.overallScore || 0),
          0
        ) / lastSubmissionsResponse.length;
      if (averageScore) {
        averageScore = averageScore.toFixed(2);
      }
      response.push({
        assignment,
        descriptions: assignment.descriptions,
        rubrics: assignment.rubrics,
        submissions: lastSubmissionsResponse,
        averageScore,
      });
    }
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

export const saveResult = async (req, res, next) => {
  try {
    const submissionResponse = await fetch(
      `https://mern-lms-saxg.onrender.com/api/assignment/get-last-submission-user-v2/${req.params.classId}/${req.params.userId}?assignmentId=${req.params.assignmentId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    const submissionsData = await submissionResponse.json();
    const mergedData = {
      ...submissionsData[0],
      descriptions: [
        ...(submissionsData[0].descriptions || []),
        ...(submissionsData[0].rubrics || []),
      ],
    };
    delete mergedData.rubrics;
    const apiKey = await Key.findById("682ca2b2bbc56ab89dbdcce5");
    const scoreResponse = await fetch(
      "https://aess-b5hr.onrender.com/api/gen-score/",
      {
        method: "POST",
        headers: {
          Authorization: `Api-Key ${apiKey.key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(mergedData),
      }
    );
    if (scoreResponse.status === 403) {
      return res.status(403).json({
        message: "API Key is not valid or expired",
      });
    }
    const { results } = await scoreResponse.json();
    if (!results || !Array.isArray(results)) {
      return res
        .status(400)
        .json({ message: "Invalid description or rubric format" });
    }
    await Promise.all(
      results.map(async (result) => {
        const {
          submission_id,
          ovr,
          scores,
          components,
          coefficients,
          feedback,
        } = result;
        await Submission.findByIdAndUpdate(
          submission_id,
          {
            $set: {
              overallAIScore: ovr,
              score: scores.length > 0 ? scores : [ovr],
              scoreComponent:
                components.length > 0 ? components : ["Overall Score"],
              coefficients: coefficients.length > 0 ? coefficients : [1],
              feedback: feedback,
            },
          },
          { new: true }
        );
      })
    );
    res.status(200).json({ results });
  } catch (error) {
    next(error);
  }
};

export const saveResults = async (req, res, next) => {
  try {
    const submissionResponse = await fetch(
      `https://mern-lms-saxg.onrender.com/api/assignment/get-last-submission-users/${req.params.classId}?assignmentId=${req.params.assignmentId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    const submissionsData = await submissionResponse.json();
    const mergedData = {
      ...submissionsData[0],
      descriptions: [
        ...(submissionsData[0].descriptions || []),
        ...(submissionsData[0].rubrics || []),
      ],
    };
    delete mergedData.rubrics;
    const apiKey = await Key.findById("682ca2b2bbc56ab89dbdcce5");
    const scoreResponse = await fetch(
      "https://aess-b5hr.onrender.com/api/gen-score/",
      {
        method: "POST",
        headers: {
          Authorization: `Api-Key ${apiKey.key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(mergedData),
      }
    );
    if (scoreResponse.status === 403) {
      return res.status(403).json({
        message: "API Key is not valid or expired",
      });
    }
    const { results } = await scoreResponse.json();
    if (!results || !Array.isArray(results)) {
      return res
        .status(400)
        .json({ message: "Invalid description or rubric format" });
    }
    await Promise.all(
      results.map(async (result) => {
        const {
          submission_id,
          ovr,
          scores,
          components,
          coefficients,
          feedback,
        } = result;
        await Submission.findByIdAndUpdate(
          submission_id,
          {
            $set: {
              overallAIScore: ovr,
              score: scores.length > 0 ? scores : [ovr],
              scoreComponent:
                components.length > 0 ? components : ["Overall Score"],
              coefficients: coefficients.length > 0 ? coefficients : [1],
              feedback: feedback,
            },
          },
          { new: true }
        );
      })
    );
    res.status(200).json({ results });
  } catch (error) {
    next(error);
  }
};

export const viewSubmissions = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const assignment = await Assignment.findById(req.params.assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    let response = [];
    if (assignment.type === "Individual") {
      const students = classItem.students;
      for (const studentId of students) {
        const studentInfo = await User.findById(studentId)
          .select("name email studentId profilePicture")
          .lean();
        const lastSubmissionInfo = await Submission.findOne({
          _id: { $in: assignment.submissions },
          uploadBy: studentId,
        })
          .sort({ createdAt: -1 })
          .lean();
        response.push({ studentInfo, lastSubmissionInfo });
      }
    } else if (assignment.type === "Group") {
      const groups = await Group.find({
        _id: { $in: assignment.groups },
      }).lean();
      for (const group of groups) {
        const groupInfo = await Group.findById(group._id)
          .populate({
            path: "members",
            model: "User",
            select: "name email studentId profilePicture",
          })
          .lean();
        const lastSubmissionInfo = await Submission.findOne({
          _id: { $in: assignment.submissions },
          groupId: groupInfo._id,
        })
          .sort({ createdAt: -1 })
          .lean();
        response.push({ groupInfo, lastSubmissionInfo });
      }
    } else {
      return res.status(400).json({ message: "Invalid assignment type" });
    }
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

export const approveAIScore = async (req, res, next) => {
  try {
    const { assignmentId } = req.params;
    const assignment = await Assignment.findById(assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    const submissions = await Submission.find({
      _id: { $in: assignment.submissions },
    });
    await Promise.all(
      submissions.map(async (submission) => {
        if (submission.groupId) {
          const group = await Group.findById(submission.groupId);
          if (!group) return;
          const individualScores = {};
          group.members.map((member) => {
            individualScores[member] =
              [...submission.score, submission.feedback] || [];
          });
          await Submission.findByIdAndUpdate(submission._id, {
            overallScore: submission.overallAIScore,
            individualScores,
          });
        } else {
          await Submission.findByIdAndUpdate(submission._id, {
            overallScore: submission.overallAIScore,
            individualScores: [...submission.score, submission.feedback] || [],
          });
        }
      })
    );
    res
      .status(200)
      .json({ message: "AI score suggestions approved successfully" });
  } catch (error) {
    next(error);
  }
};

export const getAssignmentMaxFileSize = async (req, res, next) => {
  try {
    const { assignmentId } = req.params;
    const assignment = await Assignment.findById(assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    res.status(200).json({
      maxSize: assignment.totalFileSize,
      maxNumberOfFile: assignment.maxNumberOfFile,
      maxNumberOfAttempt: assignment.maxAttempt,
    });
  } catch (error) {
    next(error);
  }
};

export const getUngroupedStudents = async (req, res, next) => {
  try {
    const { assignmentId } = req.params;
    const assignment = await Assignment.findById(assignmentId);
    if (!assignment) {
      return res.status(404).json({ message: "Assignment not found" });
    }
    const classDoc = await Class.findById(assignment.classId);
    if (!classDoc) {
      return res.status(404).json({ message: "Class not found" });
    }
    const classStudentIds = classDoc.students || [];
    const numberOfStudents = classStudentIds.length;
    const groupDocs = await Group.find({
      _id: { $in: assignment.groups || [] },
    });
    const groupedStudentIds = groupDocs.flatMap((group) => group.members || []);
    const unGroupedStudentIds = classStudentIds.filter(
      (studentId) => !groupedStudentIds.includes(studentId)
    );
    const unGroupedStudents = await User.find({
      _id: { $in: unGroupedStudentIds },
    });
    res.json({ unGroupedStudents, numberOfStudents });
  } catch (error) {
    next(error);
  }
};

schedule.scheduleJob("*/300 * * * * *", async function () {
  try {
    const res = await fetch("https://aess-b5hr.onrender.com/api/test/");
    if (res.ok) {
      console.log("Successfully pinged the server");
    }
  } catch (error) {
    console.log(error);
  }
});

schedule.scheduleJob("*/30 * * * * *", async function () {
  try {
    const now = new Date();
    const assignments = await Assignment.find({
      triggerDate: { $lt: now },
      isTrigger: false,
    });
    for (const assignment of assignments) {
      const res = await fetch(
        `https://mern-lms-saxg.onrender.com/api/assignment/save-results/${assignment.classId}/${assignment._id}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      if (res.ok) {
        await Assignment.findByIdAndUpdate(assignment._id, {
          isTrigger: true,
        });
        console.log(
          "Successfully trigger assignmentId",
          `/${assignment.classId}/${assignment._id}`
        );
      }
    }
  } catch (error) {
    console.log(error);
  }
});
