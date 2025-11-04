import { HOST } from "@/utils/constants";
import { createContext, useContext, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import {
  addMessage,
  setDirectMessagesContacts,
  setChannels,
  setSelectedChatData,
} from "@/redux/chat/chatSlice";
import { useSelector, useDispatch } from "react-redux";
import { store } from "@/redux/store";
import { setOnlineUsers } from "@/redux/user/userSlice";

const SocketContext = createContext(null);

export const useSocket = () => {
  return useContext(SocketContext);
};

export const SocketProvider = ({ children }) => {
  const socket = useRef();
  const dispatch = useDispatch();
  const { currentUser } = useSelector((state) => state.user);
  //   const { selectedChatData, selectedChatType, directMessagesContacts } = useSelector((state) => state.chat);

  useEffect(() => {
    if (currentUser) {
      // Connect to socket server
      socket.current = io(HOST, {
        withCredentials: true,
        query: { userId: currentUser._id },
      });
      socket.current.on("connect", () => {
        console.log("Connected to socket server");
      });

      const handleRecieveMessage = (message) => {
        console.log({ message }); //? debug
        //*! DOCUMENT NEEDED */
        const { selectedChatData, selectedChatType, directMessagesContacts } =
          store.getState().chat; //! Temporary solution to not re-establish the socket connection. Must not use store.getState()
        if (
          selectedChatType !== undefined &&
          (selectedChatData?._id === message.sender._id ||
            selectedChatData?._id === message.recipient._id)
        ) {
          dispatch(addMessage(message)); // If the message is from the selected chat, add it to the chat messages list
        }

        // Update the contacts list to move the most recent contact to the top
        const updatedContacts = [...directMessagesContacts];
        const contactIndex = updatedContacts.findIndex(
          (contact) =>
            contact._id === message.sender._id ||
            contact._id === message.recipient._id
        );

        if (contactIndex !== -1) {
          // Move the contact to the top
          const [recentContact] = updatedContacts.splice(contactIndex, 1);
          updatedContacts.unshift(recentContact);
        } else {
          // If the contact is not in the list, add it to the top
          const newContact = {
            _id:
              message.sender._id === currentUser._id
                ? message.recipient._id
                : message.sender._id,
            name:
              message.sender._id === currentUser._id
                ? message.recipient.name
                : message.sender.name,
            email:
              message.sender._id === currentUser._id
                ? message.recipient.email
                : message.sender.email,
            profilePicture:
              message.sender._id === currentUser._id
                ? message.recipient.profilePicture
                : message.sender.profilePicture,
          };
          updatedContacts.unshift(newContact);
        }

        dispatch(setDirectMessagesContacts(updatedContacts));
      };

      const handleRecieveChannelMessage = (message) => {
        const { selectedChatData, selectedChatType, channels } =
          store.getState().chat; //! Temporary solution to not re-establish the socket connection. Must not use store.getState()
        console.log({ channels }); //? debug
        console.log({ message }); //? debug
        if (
          selectedChatType !== undefined &&
          selectedChatData._id === message.channel._id
        ) {
          dispatch(addMessage(message));
        }
        // Update the channels list to move the most recent channel to the top
        const updatedChannels = [...channels];
        const channelIndex = updatedChannels.findIndex(
          (channel) => channel._id === message.channel._id
        );

        if (channelIndex !== -1) {
          // Move the channel to the top
          const [recentChannel] = updatedChannels.splice(channelIndex, 1);
          updatedChannels.unshift(recentChannel);
        } else {
          // If the channel is not in the list, add it to the top
          const newChannel = message.channel;
          updatedChannels.unshift(newChannel);
        }

        dispatch(setChannels(updatedChannels));
      };

      const handleChannelNameChange = (event) => {
        const { selectedChatData, selectedChatType, channels } =
          store.getState().chat; //! Temporary solution to not re-establish the socket connection. Must not use store.getState()
        console.log({ channels }); //? debug
        console.log({ event }); //? debug
        if (
          selectedChatType !== undefined &&
          selectedChatData._id === event.channel._id
        ) {
          const newChatData = { ...selectedChatData };
          newChatData.name = event.channel.name;
          const updatedChannels = channels.map((channel) =>
            channel._id === event.channel._id
              ? { ...channel, name: event.channel.name }
              : channel
          );
          dispatch(setChannels(updatedChannels));
          dispatch(setSelectedChatData(newChatData));
        }

        // Push new message to update the channels list and move the most recent channel to the top
        handleRecieveChannelMessage(event);
      };

      const handleRemovedMember = (event) => {
        const { selectedChatData, selectedChatType, channels } =
          store.getState().chat; //! Temporary solution to not re-establish the socket connection. Must not use store.getState()
        console.log({ channels }); //? debug
        console.log({ event }); //? debug
        if (
          selectedChatType !== undefined &&
          selectedChatData._id === event.channel._id
        ) {
          const newChatData = { ...selectedChatData };
          newChatData.members = event.channel.members;
          const updatedChannels = channels.map((channel) =>
            channel._id === event.channel._id
              ? { ...channel, members: event.channel.members }
              : channel
          );
          dispatch(setChannels(updatedChannels));
          dispatch(setSelectedChatData(newChatData));
        }

        // Push new message to update the channels list and move the most recent channel to the top
        handleRecieveChannelMessage(event);
      };

      const handleAddedMember = (event) => {
        const { selectedChatData, selectedChatType, channels } =
          store.getState().chat; //! Temporary solution to not re-establish the socket connection. Must not use store.getState()
        console.log({ channels }); //? debug
        console.log({ event }); //? debug
        if (
          selectedChatType !== undefined &&
          selectedChatData._id === event.channel._id
        ) {
          const newChatData = { ...selectedChatData };
          newChatData.members = event.channel.members;
          const updatedChannels = channels.map((channel) =>
            channel._id === event.channel._id
              ? { ...channel, members: event.channel.members }
              : channel
          );
          dispatch(setChannels(updatedChannels));
          dispatch(setSelectedChatData(newChatData));
        }

        // Push new message to update the channels list and move the most recent channel to the top
        handleRecieveChannelMessage(event);
      };

      socket.current.on("recieveMessage", handleRecieveMessage);
      socket.current.on("recieve-channel-message", handleRecieveChannelMessage);
      socket.current.on("changed-channel-name", handleChannelNameChange);
      socket.current.on("done-removed-member", handleRemovedMember);
      socket.current.on("done-added-member", handleAddedMember);
      socket.current.on("getOnlineUsers", (userIds) => {
        console.log("Online users: ", userIds); //? debug
        dispatch(setOnlineUsers(userIds));
      });
      socket.current.on("error", (error) => {
        console.error("Socket error:", error); //? debug
      });

      return () => {
        socket.current.disconnect(); //?
      };
    }
  }, [currentUser]); //TODO: add selectedChatData, selectedChatType,... to not re-establish the socket connection if the user is already connected

  return (
    //! Different from Zustand's Provider, the socket.current is initialized after the return statement
    <SocketContext.Provider value={socket}>{children}</SocketContext.Provider>
  );
};
