import mongoose from "mongoose";
import Channel from "../models/channel.model.js";
import User from "../models/user.model.js";

export const createChannel = async (request, response, next) => {
  try {
    const { name, members } = request.body;
    const userId = request.user.id;
    const admin = await User.findById(userId);
    if (!admin) {
      return response.status(400).send("Admin not found");
    }
    const validMembers = await User.find({ _id: { $in: members } });
    if (validMembers.length !== members.length) {
      return response.status(400).send("Some members are not valid users");
    }

    const newChannel = new Channel({ name, members, admin: userId });
    await newChannel.save();
    return response.status(201).json({ channel: newChannel });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const getChannels = async (request, response, next) => {
  try {
    const userId = new mongoose.Types.ObjectId(request.user.id);
    const channels = await Channel.find({
      $or: [{ admin: userId }, { members: userId }],
    }).sort({ updatedAt: -1 });
    return response.status(200).json({ channels });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const getChannelMessages = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    // Deep populate messages with sender data
    const channel = await Channel.findById(channelId).populate({
      path: "messages",
      populate: [
        { path: "sender", select: "name email _id profilePicture" },
        {
          path: "replyTo",
          populate: { path: "sender", select: "name email _id", model: "User" },
        },
      ],
    });
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    const messages = channel.messages;
    return response.status(200).json({ messages });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const getChannelLatestMessages = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    // Deep populate messages with sender data
    const channel = await Channel.findById(channelId).populate({
      path: "messages",
      populate: { path: "sender", select: "name email _id profilePicture" },
    });
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    const messages = channel.messages.sort((a, b) => b.timestamp - a.timestamp);
    return response.status(200).json({ message: messages[0] });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const changeChannelName = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    const channel = await Channel.findByIdAndUpdate(
      channelId,
      { $set: { name: request.body.name } },
      { new: true }
    );
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    return response.status(200).json({ channel });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const getUserInChannel = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    const channel = await Channel.findById(channelId)
      .populate("members", {
        name: 1,
        email: 1,
        profilePicture: 1,
      })
      .populate("admin", { name: 1, email: 1, profilePicture: 1 })
      .exec();
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    return response
      .status(200)
      .json({ members: channel.members.concat(channel.admin) });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const addUserToChannel = async (request, response, next) => {
  try {
    const { members } = request.body;
    const { channelId } = request.params;
    const validMembers = await User.find({ _id: { $in: members } });
    if (validMembers.length !== members.length) {
      return response.status(400).send("Some members are not valid users");
    }
    const channel = await Channel.findByIdAndUpdate(
      channelId,
      { $push: { members: { $each: members } } },
      { new: true }
    );
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    return response.status(200).json({ channel });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const removeUserFromChannel = async (request, response, next) => {
  try {
    const { members } = request.body;
    const { channelId } = request.params;
    const channel = await Channel.findByIdAndUpdate(
      channelId,
      { $pull: { members: { $in: members } } },
      { new: true }
    );
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    return response.status(200).json({ channel });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const searchUserNotInChannel = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    const { searchTerm } = request.body;

    const channel = await Channel.findById(channelId);
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    const membersInChannel = channel.members.map((member) =>
      member._id.toString()
    );
    membersInChannel.push(channel.admin.map((member) => member._id.toString()));

    if (searchTerm.trim() === "") {
      console.log("searchTerm is empty string"); //? debug
      const users = await User.find({ _id: { $nin: membersInChannel } }).select(
        {
          name: 1,
          email: 1,
          profilePicture: 1,
        }
      );
      return response.status(200).json({ users });
    } else {
      const sanitizedSearchTerm = searchTerm.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&"
      );
      const regex = new RegExp(sanitizedSearchTerm, "i");
      const users = await User.find({
        $and: [
          { _id: { $nin: membersInChannel } },
          {
            $or: [{ name: regex }, { email: regex }],
          },
        ],
      }).select({
        name: 1,
        email: 1,
        profilePicture: 1,
      });
      return response.status(200).json({ users });
    }
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const searchChannels = async (request, response, next) => {
  try {
    const { searchTerm } = request.body;
    const userId = new mongoose.Types.ObjectId(request.user.id);

    if (searchTerm.trim() === "") {
      console.log("searchTerm is empty string"); //? debug
      const channels = await Channel.find({
        $or: [{ admin: userId }, { members: userId }],
      }).sort({ updatedAt: -1 });
      return response.status(200).json({ channels });
    } else {
      const sanitizedSearchTerm = searchTerm.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&"
      );
      const regex = new RegExp(sanitizedSearchTerm, "i");
      const channels = await Channel.find({
        $and: [
          { $or: [{ admin: userId }, { members: userId }] },
          { $name: regex },
        ],
      }).sort({ updatedAt: -1 });
      return response.status(200).json({ channels });
    }
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};

export const getChannelPinnedMessages = async (request, response, next) => {
  try {
    const { channelId } = request.params;
    const channel = await Channel.findById(channelId).populate({
      path: "messages",
      populate: { path: "sender", select: "name email _id profilePicture" },
    });
    if (!channel) {
      return response.status(404).send("Channel not found");
    }
    const pinnedMessages = channel.messages.filter(
      (message) => message.isPinned
    );
    return response.status(200).json({ messages: pinnedMessages });
  } catch (error) {
    console.log({ error });
    return response.status(500).send("Internal server error");
  }
};
