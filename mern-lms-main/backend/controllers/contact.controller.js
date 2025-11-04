import mongoose from "mongoose";
import User from "../models/user.model.js";
import Message from "../models/message.model.js";

export const searchContacts = async (request, response, next) => {
  try {
    const { searchTerm } = request.body;

    if (searchTerm.trim() === "") {
      console.log("searchTerm from CONTACTS is empty string"); //? debug
      const contacts = await User.find(
        {
          _id: { $ne: request.user.id },
        },
        "-password" // Exclude the password field
      );
      return response.status(200).json({ contacts });
    } else {
      const sanitizedSearchTerm = searchTerm.replace(
        /[.*+?^${}()|[\]\\]/g,
        "\\$&"
      );
      const regex = new RegExp(sanitizedSearchTerm, "i");
      const contacts = await User.find(
        {
          $and: [
            { _id: { $ne: request.user.id } },
            {
              $or: [{ name: regex }, { email: regex }],
            },
          ],
        },
        "-password" // Exclude the password field
      );
      return response.status(200).json({ contacts });
    }
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const searchChattedContacts = async (request, response, next) => {
  try {
    const { searchTerm } = request.query; // Get the search term from the query parameters
    const userId = new mongoose.Types.ObjectId(request.user.id);

    // Sanitize the search term to prevent regex injection
    const sanitizedSearchTerm = searchTerm
      ? searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      : "";
    const regex = new RegExp(sanitizedSearchTerm, "i");

    // Find all contacts the user has chatted with
    const contacts = await Message.aggregate([
      {
        $match: {
          $or: [{ sender: userId }, { recipient: userId }],
        },
      },
      {
        $group: {
          _id: {
            $cond: {
              if: { $eq: ["$sender", userId] },
              then: "$recipient",
              else: "$sender",
            },
          },
        },
      },
      {
        $lookup: {
          from: "users",
          localField: "_id",
          foreignField: "_id",
          as: "contactInfo",
        },
      },
      {
        $unwind: "$contactInfo",
      },
      {
        $match: {
          $or: [{ "contactInfo.name": regex }, { "contactInfo.email": regex }],
        },
      },
      {
        $project: {
          _id: 1,
          name: "$contactInfo.name",
          email: "$contactInfo.email",
          profilePicture: "$contactInfo.profilePicture",
        },
      },
      {
        $sort: { name: 1 }, // Sort alphabetically by name
      },
    ]);

    return response.status(200).json({ contacts });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const getContactsForDMList = async (request, response, next) => {
  try {
    let userId = request.user.id;
    userId = new mongoose.Types.ObjectId(userId);

    const contacts = await Message.aggregate([
      {
        $match: {
          $or: [{ sender: userId }, { recipient: userId }],
        },
      },
      {
        $sort: { timestamp: -1 },
      },
      {
        $group: {
          _id: {
            $cond: {
              if: { $eq: ["$sender", userId] },
              then: "$recipient",
              else: "$sender",
            },
          },
          lastMessageTime: { $first: "$timestamp" },
        },
      },
      {
        $lookup: {
          from: "users",
          localField: "_id",
          foreignField: "_id",
          as: "contactInfo",
        },
      },
      {
        $unwind: "$contactInfo",
      },
      {
        $project: {
          _id: 1,
          lastMessageTime: 1,
          email: "$contactInfo.email",
          name: "$contactInfo.name",
          profilePicture: "$contactInfo.profilePicture",
        },
      },
      {
        $sort: { lastMessageTime: -1 },
      },
    ]);

    return response.status(200).json({ contacts });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};

export const getAllContact = async (request, response, next) => {
  try {
    const users = await User.find(
      {
        _id: { $ne: request.user.id },
      },
      "name email _id"
    );

    const contacts = users.map((user) => ({
      label: user.name ? user.name : user.email,
      value: user._id,
    }));

    return response.status(200).json({ contacts });
  } catch (error) {
    console.log(error);
    return response.status(500).send("Internal server error");
  }
};
