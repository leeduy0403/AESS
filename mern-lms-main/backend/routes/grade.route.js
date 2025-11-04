import express from "express";
import { verifyToken } from "../utils/verifyUser.js";

const router = express.Router();

router.get("/get-grade", verifyToken, getGrade);

export default router;
