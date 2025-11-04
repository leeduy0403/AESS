import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createSubmission,
  deleteSubmission,
  getSubmissions,
  getSubmissionsOfAssignmentOfUser,
  updateMemberScores,
  updateSubmission,
} from "../controllers/submission.controller.js";

const router = express.Router();

router.get("/get/:assignmentId", verifyToken, getSubmissions);
router.get(
  "/get-user-submissions/:assignmentId/:userId",
  verifyToken,
  getSubmissionsOfAssignmentOfUser
);
router.post("/create/:assignmentId", verifyToken, createSubmission);
router.delete(
  "/delete/:assignmentId/:submissionId",
  verifyToken,
  deleteSubmission
);
router.put(
  "/update/:assignmentId/:submissionId",
  verifyToken,
  updateSubmission
);
router.put("/update-overallScores", verifyToken, updateMemberScores);

export default router;
