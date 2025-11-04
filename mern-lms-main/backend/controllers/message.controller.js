import Message from "../models/message.model.js";
import { mkdirSync, renameSync } from "fs";

export const getMessages = async (request, response, next) => {
  try {
    // console.log({request});
    const user1 = request.user.id;
    const user2 = request.body.id;
    if (!user1 || !user2) {
      return response.status(400).send("Both user ID's are required");
    }

    const messages = await Message.find({
      $or: [
        { sender: user1, recipient: user2 },
        { sender: user2, recipient: user1 },
      ],
    })
      .sort({ timestamp: 1 })
      .populate({
        path: "replyTo",
        populate: {
          path: "sender",
          select: "name email _id",
          model: "User",
        },
      });

    return response.status(200).json({ messages });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const uploadFile = async (request, response, next) => {
  try {
    if (!request.file) {
      return response.status(400).send("File is required");
    }
    const date = Date.now();
    let fileDir = `uploads/files/${date}`;
    let fileName = `${fileDir}/${request.file.originalname}`;

    // Store files locally
    // mkdirSync(fileDir, { recursive: true });
    // renameSync(request.file.path, fileName);

    return response.status(200).json({ filePath: fileName });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const getLatestMessages = async (request, response, next) => {
  try {
    // console.log({request});
    const user1 = request.user.id;
    const user2 = request.body.id;
    if (!user1 || !user2) {
      return response.status(400).send("Both user ID's are required");
    }

    const messages = await Message.find({
      $or: [
        { sender: user1, recipient: user2 },
        { sender: user2, recipient: user1 },
      ],
    }).sort({ timestamp: -1 });

    return response.status(200).json({ message: messages[0] });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const pinMessage = async (request, response, next) => {
  try {
    const { messageId } = request.body;
    const message = await Message.findByIdAndUpdate(
      messageId,
      { $set: { isPinned: true } },
      { new: true }
    );
    if (!message) {
      return response.status(404).send("Message not found");
    }
    return response.status(200).json({ message });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const unpinMessage = async (request, response, next) => {
  try {
    const { messageId } = request.body;
    const message = await Message.findByIdAndUpdate(
      messageId,
      { $set: { isPinned: false } },
      { new: true }
    );
    if (!message) {
      return response.status(404).send("Message not found");
    }
    return response.status(200).json({ message });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const getPinnedMessages = async (request, response, next) => {
  try {
    const user1 = request.user.id;
    const user2 = request.body.id;
    if (!user1 || !user2) {
      return response.status(400).send("Both user ID's are required");
    }

    const messages = await Message.find({
      $and: [
        {
          $or: [
            { sender: user1, recipient: user2 },
            { sender: user2, recipient: user1 },
          ],
        },
        { isPinned: true },
      ],
    })
      .sort({ timestamp: 1 })
      .populate({ path: "sender", select: "name email _id profilePicture" });
    return response.status(200).json({ messages });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};
