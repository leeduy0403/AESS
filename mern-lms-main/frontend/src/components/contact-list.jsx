import { useSelector, useDispatch } from "react-redux";
import {
  setSelectedChatData,
  setSelectedChatType,
  setSelectedChatMessages,
} from "@/redux/chat/chatSlice";
import { Avatar, AvatarImage } from "./ui/avatar";
import { apiClient } from "@/lib/api-client";
import {
  GET_LATEST_MESSAGES_ROUTE,
  GET_CHANNEL_LATEST_MESSAGES_ROUTE,
} from "@/utils/constants.js";
import { useState, useEffect } from "react";

function ContactList({ contacts, isChannel = false }) {
  //   console.log("CONTACT LIST RENDERED"); //? debug
  const { selectedChatData } = useSelector((state) => state.chat);
  const { onlineUsers, currentUser } = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const [latestMessage, setLatestMessage] = useState({});
  const [channelLatestMessage, setChannelLatestMessage] = useState({});
  const [messageLoading, setMessageLoading] = useState(true);

  const handleClick = (contact) => {
    if (isChannel) {
      dispatch(setSelectedChatType("channel"));
    } else {
      dispatch(setSelectedChatType("contact"));
    }
    dispatch(setSelectedChatData(contact));
    // If select another contact
    if (selectedChatData && selectedChatData._id !== contact?._id) {
      // Clear message screen before render another chat
      dispatch(setSelectedChatMessages([]));
    }
  };

  useEffect(() => {
    const getLatestDM = async (contactId) => {
      try {
        const response = await apiClient.post(
          GET_LATEST_MESSAGES_ROUTE,
          { id: contactId },
          { withCredentials: true }
        );
        // console.log(contactId, response); //? debug
        if (response.data.message) {
          setLatestMessage((latestMessage) => ({
            ...latestMessage,
            [contactId]: response.data.message,
          }));
        } else {
          setLatestMessage((latestMessage) => ({
            ...latestMessage,
            [contactId]: {},
          }));
        }
      } catch (error) {
        console.log({ error });
      }
    };

    const getChannelLatestMessage = async (channelId) => {
      try {
        const response = await apiClient.get(
          `${GET_CHANNEL_LATEST_MESSAGES_ROUTE}/${channelId}`,
          { withCredentials: true }
        );
        //   console.log(channelId, response); //? debug
        if (response.data.message) {
          setChannelLatestMessage((channelLatestMessage) => ({
            ...channelLatestMessage,
            [channelId]: response.data.message,
          }));
        } else {
          setChannelLatestMessage((channelLatestMessage) => ({
            ...channelLatestMessage,
            [channelId]: null,
          }));
        }
      } catch (error) {
        console.log({ error });
      }
    };

    setMessageLoading(true);
    if (isChannel) {
      contacts.forEach((contact) => {
        getChannelLatestMessage(contact?._id);
      });
      setMessageLoading(false);
    } else {
      contacts.forEach((contact) => {
        getLatestDM(contact?._id);
      });
      setMessageLoading(false);
    }
  }, [contacts]);

  return (
    <>
      {contacts?.length <= 0 && (
        <div className="flex flex-col h-full w-full items-center justify-center mt-10">
          <h3 className="font-semibold">
            {isChannel ? "No group" : "No contact"}
          </h3>
        </div>
      )}
      <div className="mt-0 overflow-hidden">
        {contacts?.length > 0 &&
          contacts.map((contact) => (
            <div
              key={contact?._id}
              className={`cursor-pointer py-2 pl-10 transition-all duration-100 ${
                selectedChatData && selectedChatData._id === contact?._id
                  ? "bg-[#DCDFE4] hover:bg-[#b3b5b9]"
                  : "hover:bg-[#b3b5b9]"
              }`}
              onClick={() => handleClick(contact)}
            >
              <div className="flex items-center justify-start gap-5 text-black font-semibold max-w-[100%]">
                {!isChannel && (
                  <div className="relative flex-shrink-0">
                    <Avatar className="h-10 w-10 overflow-hidden rounded-full border-black border">
                      <AvatarImage
                        src={contact?.profilePicture}
                        alt="profile"
                        className="h-full w-full bg-white object-cover"
                      />
                    </Avatar>
                    <div
                      className={`rounded-full w-3 h-3 ml-2 mt-2 absolute bottom-0 right-0 ${
                        onlineUsers?.includes(contact?._id)
                          ? "bg-green-500"
                          : "bg-red-500"
                      }`}
                    ></div>
                  </div>
                )}
                {isChannel && (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#ffffff22] border border-black flex-shrink-0">
                    #
                  </div>
                )}
                {isChannel ? (
                  <div className="flex flex-col w-full">
                    <span className="max-w-[50%]">{contact?.name}</span>
                    <span
                      className={`${
                        messageLoading
                          ? "text-transparent invisible"
                          : "text-[#00000099] visible"
                      } h-7 pr-4 overflow-hidden font-normal transition-all duration-200 truncate max-w-[85%]`}
                    >
                      {channelLatestMessage[contact?._id]
                        ? channelLatestMessage[contact?._id].sender._id ===
                          currentUser._id
                          ? "You" +
                            ": " +
                            channelLatestMessage[contact?._id].content
                          : channelLatestMessage[contact?._id].sender.name +
                            ": " +
                            channelLatestMessage[contact?._id].content
                        : ""}
                    </span>
                  </div>
                ) : (
                  <div className="flex flex-col w-full">
                    <span className="max-w-[50%]">
                      {contact?.name ? contact?.name : contact?.email}
                    </span>
                    <span
                      className={`${
                        messageLoading
                          ? "text-transparent invisible"
                          : "text-[#00000099] visible"
                      } h-7 pr-4 overflow-hidden font-normal transition-all duration-200 truncate max-w-[85%]`}
                    >
                      {latestMessage[contact?._id]
                        ? latestMessage[contact?._id].sender === currentUser._id
                          ? "You" + ": " + latestMessage[contact?._id].content
                          : latestMessage[contact?._id].content
                        : ""}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
      </div>
    </>
  );
}

export default ContactList;
