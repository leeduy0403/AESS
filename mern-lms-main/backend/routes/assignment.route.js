import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  approveAIScore,
  createAssignment,
  deleteAssignment,
  getAssignments,
  getLastSubmissionOfAssignmentsOfUser,
  getLastSubmissionOfAssignmentsOfUsers,
  getOngoingAssignments,
  saveResults,
  updateAssignment,
  viewSubmissions,
  getAssignmentMaxFileSize,
  getUngroupedStudents,
  getLastSubmissionOfAssignmentsOfUserV2,
  saveResult,
} from "../controllers/assignment.controller.js";

const router = express.Router();

router.get("/get/:classId", verifyToken, getAssignments);
router.get(
  "/get-ongoing-assignments/:classId",
  verifyToken,
  getOngoingAssignments
);
router.get(
  "/get-last-submission-user/:classId/:userId",
  verifyToken,
  getLastSubmissionOfAssignmentsOfUser
);
router.get(
  "/get-last-submission-user-v2/:classId/:userId",
  getLastSubmissionOfAssignmentsOfUserV2
);
router.get(
  "/get-last-submission-users/:classId",
  getLastSubmissionOfAssignmentsOfUsers
);
router.post("/save-result/:classId/:assignmentId/:userId", saveResult);
router.post("/save-results/:classId/:assignmentId", saveResults);
router.get("/view-submissions/:classId/:assignmentId", viewSubmissions);
router.post("/approve/:assignmentId", approveAIScore);
router.post("/create/:classId", verifyToken, createAssignment);
router.delete("/delete/:classId/:assignmentId", verifyToken, deleteAssignment);
router.put("/update/:classId/:assignmentId", verifyToken, updateAssignment);
router.get(
  "/get-assignment-max-file-size/:assignmentId",
  verifyToken,
  getAssignmentMaxFileSize
);
router.get(
  "/get-ungrouped-students/:assignmentId",
  verifyToken,
  getUngroupedStudents
);

export default router;
