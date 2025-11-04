import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createGroup,
  deleteGroup,
  getGroupOfAssignmentOfUser,
  getGroupsOfAssignment,
  joinGroup,
  leaveGroup,
  removeMemberOfGroup,
  updateGroup,
} from "../controllers/group.controller.js";

const router = express.Router();

router.get("/get/:assignmentId", verifyToken, getGroupsOfAssignment);
router.get(
  "/get-user-group/:assignmentId/:userId",
  verifyToken,
  getGroupOfAssignmentOfUser
);
router.post("/create/:assignmentId", verifyToken, createGroup);
router.post("/join/:assignmentId", verifyToken, joinGroup);
router.post("/leave/:assignmentId", verifyToken, leaveGroup);
router.post(
  "/remove-member/:classId/:assignmentId",
  verifyToken,
  removeMemberOfGroup
);
router.delete("/delete/:assignmentId/:groupId", verifyToken, deleteGroup);
router.put("/update/:assignmentId/:groupId", verifyToken, updateGroup);

export default router;
