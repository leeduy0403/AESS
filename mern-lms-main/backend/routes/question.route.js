import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createQuestion,
  createQuestionFromSubmission,
  deleteQuestion,
  editQuestion,
  getQuestions,
} from "../controllers/question.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createQuestion);
router.post(
  "/create-from-submission",
  verifyToken,
  createQuestionFromSubmission
);
router.put("/edit/:questionId", verifyToken, editQuestion);
router.delete("/delete/:topicId/:questionId", verifyToken, deleteQuestion);
router.get("/get-questions/:topicId", verifyToken, getQuestions);

export default router;
