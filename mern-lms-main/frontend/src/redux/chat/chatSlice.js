import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  selectedChatType: undefined,
  selectedChatData: undefined,
  selectedChatMessages: [],
  directMessagesContacts: [],
  channels: [],
  replyingMessage: undefined,
};

export const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    setReplyingMessage: (state, action) => {
      state.replyingMessage = action.payload;
    },
    addChannel: (state, action) => {
      state.channels.unshift(action.payload); // Add channel to the start
    },
    setChannels: (state, action) => {
      state.channels = action.payload;
    },
    setSelectedChatType: (state, action) => {
      state.selectedChatType = action.payload;
    },
    setSelectedChatData: (state, action) => {
      state.selectedChatData = action.payload;
    },
    setSelectedChatMessages: (state, action) => {
      state.selectedChatMessages = action.payload;
    },
    setDirectMessagesContacts: (state, action) => {
      state.directMessagesContacts = action.payload;
    },
    closeChat: (state) => {
      state.selectedChatType = undefined;
      state.selectedChatData = undefined;
      state.selectedChatMessages = [];
    },
    addMessage: (state, action) => {
      const message = action.payload;
      state.selectedChatMessages.push({
        ...message,
        recipient:
          state.selectedChatType === "channel"
            ? message.recipient
            : message.recipient._id,
        sender:
          state.selectedChatType === "channel"
            ? message.sender
            : message.sender._id,
      });
    },
  },
});

// Action creators are generated for each case reducer function
export const {
  setSelectedChatType,
  setSelectedChatData,
  setSelectedChatMessages,
  setDirectMessagesContacts,
  closeChat,
  addMessage,
  setChannels,
  addChannel,
  setReplyingMessage,
} = chatSlice.actions;

export default chatSlice.reducer;
