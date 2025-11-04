import { Router } from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  getMessages,
  uploadFile,
  getLatestMessages,
  pinMessage,
  getPinnedMessages,
  unpinMessage,
} from "../controllers/message.controller.js";

const messageRoute = Router();

messageRoute.post("/get-messages", verifyToken, getMessages);
messageRoute.post("/upload-file", verifyToken, uploadFile);

messageRoute.post("/get-latest-message", verifyToken, getLatestMessages);
messageRoute.post("/pin-message", verifyToken, pinMessage);
messageRoute.post("/unpin-message", verifyToken, unpinMessage);
messageRoute.post("/get-pinned-messages", verifyToken, getPinnedMessages);

export default messageRoute;
