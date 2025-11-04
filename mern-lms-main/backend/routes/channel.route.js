import { Router } from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createChannel,
  getChannels,
  getChannelMessages,
  changeChannelName,
  addUserToChannel,
  removeUserFromChannel,
  getChannelLatestMessages,
  getUserInChannel,
  searchUserNotInChannel,
  searchChannels,
  getChannelPinnedMessages,
} from "../controllers/channel.controller.js";

const channelRoute = Router();

channelRoute.post("/create-channel", verifyToken, createChannel);
channelRoute.get("/get-channel", verifyToken, getChannels);
channelRoute.post("/search-channel", verifyToken, searchChannels);
channelRoute.get(
  "/get-channel-messages/:channelId",
  verifyToken,
  getChannelMessages
);
channelRoute.post(
  "/change-channel-name/:channelId",
  verifyToken,
  changeChannelName
);
channelRoute.post(
  "/add-user-to-channel/:channelId",
  verifyToken,
  addUserToChannel
);
channelRoute.post(
  "/remove-user-from-channel/:channelId",
  verifyToken,
  removeUserFromChannel
);
channelRoute.get(
  "/get-channel-latest-message/:channelId",
  verifyToken,
  getChannelLatestMessages
);
channelRoute.get(
  "/get-user-in-channel/:channelId",
  verifyToken,
  getUserInChannel
);
channelRoute.post(
  "/search-user-not-in-channel/:channelId",
  verifyToken,
  searchUserNotInChannel
);
channelRoute.get(
  "/get-channel-pinned-messages/:channelId",
  verifyToken,
  getChannelPinnedMessages
);

export default channelRoute;
