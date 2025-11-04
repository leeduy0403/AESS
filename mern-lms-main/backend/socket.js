import { Server as SocketIOServer } from "socket.io";
import Message from "./models/message.model.js";
import Channel from "./models/channel.model.js";
import User from "./models/user.model.js";

const setupSocket = (server) => {
  const io = new SocketIOServer(server, {
    cors: {
      origin: process.env.ORIGIN,
      methods: ["GET", "POST"],
      credentials: true,
    },
  });

  // Store users id and correspond socket id
  const userSocketMap = new Map();

  const disconnect = (socket) => {
    console.log(`Client disconnected: ${socket.id}, at ${Date.now()}`);
    for (const [userId, socketId] of userSocketMap.entries()) {
      if (socketId === socket.id) {
        userSocketMap.delete(userId);
        break;
      }
    }
    io.emit("getOnlineUsers", [...userSocketMap.keys()]);
  };

  const sendMessage = async (message) => {
    const senderSocketId = userSocketMap.get(message.sender);
    const recipientSocketId = userSocketMap.get(message.recipient);

    const createdMessage = await Message.create(message);

    // Search message by id and embed the sender and recipient data inside
    const messageData = await Message.findById(createdMessage._id)
      .populate("sender", "id email name profilePicture")
      .populate("recipient", "id email name profilePicture")
      .populate({
        path: "replyTo",
        populate: {
          path: "sender",
          select: "name email _id",
          model: "User",
        },
      });

    if (recipientSocketId) {
      io.to(recipientSocketId).emit("recieveMessage", messageData);
      console.log("Message sent to recipient at socket: ", recipientSocketId);
    }
    if (senderSocketId) {
      io.to(senderSocketId).emit("recieveMessage", messageData);
      console.log("Message sent to sender at socket: ", senderSocketId);
    }
  };

  const sendChannelMessage = async (message) => {
    const { channelId, sender, content, messageType, fileUrl, replyTo } =
      message;

    const createdMessage = await Message.create({
      sender,
      recipient: null,
      content,
      messageType,
      replyTo,
      timestamp: Date.now(),
      fileUrl,
    });

    // Create new message and embed the sender data
    const messageData = await Message.findById(createdMessage._id)
      .populate("sender", "id email name profilePicture")
      .populate({
        path: "replyTo",
        populate: {
          path: "sender",
          select: "name email _id",
          model: "User",
        },
      })
      .exec();

    // Push newly created message into channel's list of messages
    await Channel.findByIdAndUpdate(channelId, {
      $push: { messages: createdMessage._id },
    });

    const channel = await Channel.findById(channelId).populate("members");

    const finalData = {
      ...messageData._doc,
      //   channelId: channel._id,
      channel: channel,
    };

    if (channel && channel.members) {
      channel.members.forEach((member) => {
        // Emit to user in the channel that is online
        const memberSocketId = userSocketMap.get(member._id.toString());
        if (memberSocketId) {
          io.to(memberSocketId).emit("recieve-channel-message", finalData);
          console.log(
            `Message sent to member ${member._id} at socket: ${memberSocketId}`
          );
        }
      });
      // Emit to all admins that are online in the channel
      channel.admin.forEach((admin) => {
        const adminSocketId = userSocketMap.get(admin._id.toString());
        if (adminSocketId) {
          io.to(adminSocketId).emit("recieve-channel-message", finalData);
          console.log(
            `Message sent to admin ${admin._id} at socket: ${adminSocketId}`
          );
        }
      });
    }
  };

  const changeChannelNameHandler = async (event) => {
    try {
      const { channelId, sender, name } = event;
      const channel = await Channel.findByIdAndUpdate(
        channelId,
        { $set: { name: name } },
        { new: true }
      );
      if (!channel) {
        const senderSocketId = userSocketMap.get(sender._id.toString());
        io.to(senderSocketId).emit("error", {
          message: "Error changing channel name",
        });
        return;
      }
      const createdMessage = await Message.create({
        sender: sender._id,
        recipient: null,
        content: `${sender.name} changed the group name to "${name}"`,
        messageType: "event",
        timestamp: Date.now(),
        fileUrl: null,
      });

      // Create new message and embed the sender data
      const messageData = await Message.findById(createdMessage._id)
        .populate("sender", "id email name profilePicture")
        .exec();

      // Push newly created message into channel's list of messages
      await Channel.findByIdAndUpdate(channelId, {
        $push: { messages: createdMessage._id },
      });

      const finalData = {
        ...messageData._doc,
        //   channelId: channel._id,
        channel: channel,
      };

      if (channel && channel.members) {
        channel.members.forEach((member) => {
          // Emit to user in the channel that is online
          const memberSocketId = userSocketMap.get(member._id.toString());
          if (memberSocketId) {
            io.to(memberSocketId).emit("changed-channel-name", finalData);
            console.log(
              `Message sent to member ${member._id} at socket: ${memberSocketId}`
            );
          }
        });
        // Emit to all admins that are online in the channel
        channel.admin.forEach((admin) => {
          const adminSocketId = userSocketMap.get(admin._id.toString());
          if (adminSocketId) {
            io.to(adminSocketId).emit("changed-channel-name", finalData);
            console.log(
              `Message sent to admin ${admin._id} at socket: ${adminSocketId}`
            );
          }
        });
      }
    } catch (error) {
      console.log({ error });
      io.emit("error", {
        message: "Failed to change channel name",
      });
    }
  };

  const removeMemberChannelHandler = async (event) => {
    try {
      const { channelId, sender, memberId, memberName } = event;
      const channel = await Channel.findByIdAndUpdate(
        channelId,
        { $pull: { members: { $in: memberId } } },
        { new: true }
      );
      if (!channel) {
        const senderSocketId = userSocketMap.get(sender._id.toString());
        io.to(senderSocketId).emit("error", {
          message: "Error removing member from channel",
        });
        return;
      }
      const content =
        sender._id === memberId[0]
          ? `${memberName} left the group`
          : `${sender.name} removed ${memberName} from the group`;
      const createdMessage = await Message.create({
        sender: sender._id,
        recipient: null,
        content: content,
        messageType: "event",
        timestamp: Date.now(),
        fileUrl: null,
      });

      // Create new message and embed the sender data
      const messageData = await Message.findById(createdMessage._id)
        .populate("sender", "id email name profilePicture")
        .exec();

      // Push newly created message into channel's list of messages
      await Channel.findByIdAndUpdate(channelId, {
        $push: { messages: createdMessage._id },
      });

      const finalData = {
        ...messageData._doc,
        //   channelId: channel._id,
        channel: channel,
      };

      if (channel && channel.members) {
        channel.members.forEach((member) => {
          // Emit to user in the channel that is online
          const memberSocketId = userSocketMap.get(member._id.toString());
          if (memberSocketId) {
            io.to(memberSocketId).emit("done-removed-member", finalData);
            console.log(
              `Message sent to member ${member._id} at socket: ${memberSocketId}`
            );
          }
        });
        // Emit to all admins that are online in the channel
        channel.admin.forEach((admin) => {
          const adminSocketId = userSocketMap.get(admin._id.toString());
          if (adminSocketId) {
            io.to(adminSocketId).emit("done-removed-member", finalData);
            console.log(
              `Message sent to admin ${admin._id} at socket: ${adminSocketId}`
            );
          }
        });
      }
      const removedUserSocketId = userSocketMap.get(memberId.toString());
      if (removedUserSocketId) {
        io.to(removedUserSocketId).emit("done-removed-member", finalData);
        console.log(
          `Message sent to REMOVED user ${memberId} at socket: ${removedUserSocketId}`
        );
      }
    } catch (error) {
      console.log({ error });
      io.emit("error", {
        message: "Failed to remove member from channel",
      });
    }
  };

  const addMemberChannelHandler = async (event) => {
    try {
      const { channelId, sender, memberId, memberName } = event; // memberId is an ARRAY of ids, memberName is an ARRAY of names
      const validMembers = await User.find({ _id: { $in: memberId } });
      if (validMembers.length !== memberId.length) {
        const senderSocketId = userSocketMap.get(sender._id.toString());
        io.to(senderSocketId).emit("error", {
          message: "Some members are not valid users",
        });
        return;
      }
      const channel = await Channel.findByIdAndUpdate(
        channelId,
        { $push: { members: { $each: memberId } } },
        { new: true }
      );
      if (!channel) {
        const senderSocketId = userSocketMap.get(sender._id.toString());
        io.to(senderSocketId).emit("error", {
          message: "Error adding members to channel",
        });
        return;
      }
      memberName.forEach(async (name) => {
        const content = `${sender.name} added ${name} to the group`;
        const createdMessage = await Message.create({
          sender: sender._id,
          recipient: null,
          content: content,
          messageType: "event",
          timestamp: Date.now(),
          fileUrl: null,
        });

        // Create new message and embed the sender data
        const messageData = await Message.findById(createdMessage._id)
          .populate("sender", "id email name profilePicture")
          .exec();

        // Push newly created message into channel's list of messages
        await Channel.findByIdAndUpdate(channelId, {
          $push: { messages: createdMessage._id },
        });

        const finalData = {
          ...messageData._doc,
          //   channelId: channel._id,
          channel: channel,
        };

        if (channel && channel.members) {
          channel.members.forEach((member) => {
            // Emit to user in the channel that is online
            const memberSocketId = userSocketMap.get(member._id.toString());
            if (memberSocketId) {
              io.to(memberSocketId).emit("done-added-member", finalData);
              console.log(
                `Message sent to member ${member._id} at socket: ${memberSocketId}`
              );
            }
          });
          // Emit to all admins that are online in the channel
          channel.admin.forEach((admin) => {
            const adminSocketId = userSocketMap.get(admin._id.toString());
            if (adminSocketId) {
              io.to(adminSocketId).emit("done-added-member", finalData);
              console.log(
                `Message sent to admin ${admin._id} at socket: ${adminSocketId}`
              );
            }
          });
        }
      });
    } catch (error) {
      console.log({ error });
      io.emit("error", {
        message: "Failed to add members to channel",
      });
    }
  };

  io.on("connection", (socket) => {
    const userId = socket.handshake.query.userId;

    if (userId) {
      if (userSocketMap.has(userId)) {
        const oldSocketId = userSocketMap.get(userId);
        console.log(`user re-connected: ${userId} with id: ${oldSocketId}`);
      } else {
        userSocketMap.set(userId, socket.id);
        console.log(`user connected: ${userId} with id: ${socket.id}`);
      }
    } else {
      console.log("userId not provided");
    }

    console.log("Online users: ", [...userSocketMap.keys()]); //? debug
    io.emit("getOnlineUsers", [...userSocketMap.keys()]); // emit online users to all clients

    socket.on("send-message", sendMessage);
    socket.on("send-channel-message", sendChannelMessage);
    socket.on("change-channel-name", changeChannelNameHandler); // listen for change channel name event
    socket.on("remove-member", removeMemberChannelHandler); // listen for change channel name event
    socket.on("add-members", addMemberChannelHandler); // listen for change channel name event
    socket.on("disconnect", () => disconnect(socket)); // delete userId from userSocketMap when user disconnects
  });
};

export default setupSocket;
