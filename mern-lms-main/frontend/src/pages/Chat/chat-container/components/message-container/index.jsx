import { apiClient } from "@/lib/api-client";
import {
  GET_ALL_MESSAGES_ROUTE,
  GET_CHANNEL_MESSAGES,
  PIN_MESSAGE_ROUTE,
  UNPIN_MESSAGE_ROUTE,
} from "@/utils/constants";
import moment from "moment";
import { useEffect, useRef, useState } from "react";
import { MdAttachFile } from "react-icons/md";
import { IoArrowDownOutline, IoCloseSharp } from "react-icons/io5";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import ForwardMessage from "./forward-message/index.jsx";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  setSelectedChatMessages,
  setReplyingMessage,
} from "@/redux/chat/chatSlice";
import { useSelector, useDispatch } from "react-redux";
import { RiReplyFill, RiPushpin2Fill, RiUnpinFill } from "react-icons/ri";

function MessageContainer({ showInfo, onReplyMessageClick }) {
  const scrollRef = useRef(null);
  const dispatch = useDispatch();
  const { selectedChatType, selectedChatData, selectedChatMessages } =
    useSelector((state) => state.chat);
  const { currentUser, onlineUsers } = useSelector((state) => state.user);

  const [showImage, setShowImage] = useState(false);
  const [imageURL, setImageURL] = useState(null);

  useEffect(() => {
    const getMessages = async () => {
      try {
        const response = await apiClient.post(
          GET_ALL_MESSAGES_ROUTE,
          { id: selectedChatData._id },
          { withCredentials: true }
        );
        if (response.data.messages) {
          dispatch(setSelectedChatMessages(response.data.messages));
        }
      } catch (error) {
        console.log({ error });
      }
    };

    const getChannelMessages = async () => {
      try {
        const response = await apiClient.get(
          `${GET_CHANNEL_MESSAGES}/${selectedChatData._id}`,
          { withCredentials: true }
        );
        if (response.data.messages) {
          dispatch(setSelectedChatMessages(response.data.messages));
        }
      } catch (error) {
        console.log({ error });
      }
    };

    if (selectedChatData._id) {
      if (selectedChatType === "contact") {
        getMessages();
      } else if (selectedChatType === "channel") {
        getChannelMessages();
      }
    }
  }, [selectedChatData, selectedChatType, dispatch]);

  // Scroll to the end
  useEffect(() => {
    const container = scrollRef.current?.parentNode;
    if (container) {
      container.scrollTo({
        top: container.scrollHeight,
        // behavior: "smooth",
      });
    }
  }, [selectedChatMessages]);

  const checkIfImage = (filePath) => {
    const imageRegex =
      /\.(jpg|jpeg|png|gif|bmp|tiff|tif|webp|svg|ico|heic|heif)$/i;
    return imageRegex.test(filePath);
  };

  const groupMessages = (messages) => {
    const groups = [];
    let currentGroup = [];
    let lastDate = null;
    let lastSender = null;
    let lastTimestamp = null;

    messages.forEach((message, index) => {
      const messageDate = moment(message.timestamp).format("YYYY-MM-DD");
      if (message.messageType === "event") {
        // Push the current group if it has messages
        if (currentGroup.length > 0) {
          groups.push({
            date: lastDate,
            messages: [...currentGroup],
            showDate:
              groups.length === 0 ||
              lastDate !== groups[groups.length - 1].date,
          });
          currentGroup = [];
        }

        groups.push({
          date: messageDate,
          messages: [message],
          showDate: groups.length === 0 || messageDate !== lastDate,
        });
        lastDate = messageDate;
      } else {
        const currentSender =
          selectedChatType === "contact" ? message.sender : message.sender._id;

        // Check if this is a new date
        const isNewDate = messageDate !== lastDate;

        // Check if this is a new sender
        const isNewSender = lastSender !== currentSender;

        // Check if more than 1 minute has passed since the last message
        const isNewTimeGroup =
          lastTimestamp &&
          moment(message.timestamp).diff(moment(lastTimestamp), "minutes") >= 1;

        // If any of these conditions are true, start a new group
        if (
          isNewDate ||
          isNewSender ||
          isNewTimeGroup ||
          currentGroup.length === 0
        ) {
          if (currentGroup.length > 0) {
            groups.push({
              date: lastDate,
              messages: [...currentGroup],
              showDate:
                groups.length === 0 ||
                lastDate !== groups[groups.length - 1].date,
            });
            currentGroup = [];
          }

          // If it's a new date, mark it to show the date separator
          if (isNewDate) {
            lastDate = messageDate;
          }
        }

        currentGroup.push(message);
        lastSender = currentSender;
        lastTimestamp = message.timestamp;

        // If this is the last message, add the current group
        if (index === messages.length - 1) {
          groups.push({
            date: lastDate,
            messages: [...currentGroup],
            showDate:
              groups.length === 0 ||
              lastDate !== groups[groups.length - 1].date,
          });
        }
      }
    });

    return groups;
  };

  const pinMessage = async (message) => {
    const response = await apiClient.post(
      PIN_MESSAGE_ROUTE,
      { messageId: message._id },
      { withCredentials: true }
    );
    if (response.data.message) {
      const messages = [...selectedChatMessages];
      const updatedMessages = messages.map((msg) => {
        if (msg._id === message._id) {
          return { ...msg, isPinned: true };
        }
        return msg;
      });
      dispatch(setSelectedChatMessages(updatedMessages));
      const container = scrollRef.current?.parentNode;
      if (container) {
        const currentScrollTop = container.scrollTop;
        setTimeout(() => {
          container.scrollTo({
            top: currentScrollTop,
            behavior: "auto",
          });
        }, 0);
      }
    }
  };

  const unpinMessage = async (message) => {
    const response = await apiClient.post(
      UNPIN_MESSAGE_ROUTE,
      { messageId: message._id },
      { withCredentials: true }
    );
    if (response.data.message) {
      const messages = [...selectedChatMessages];
      const updatedMessages = messages.map((msg) => {
        if (msg._id === message._id) {
          return { ...msg, isPinned: false };
        }
        return msg;
      });
      dispatch(setSelectedChatMessages(updatedMessages));
      const container = scrollRef.current?.parentNode;
      if (container) {
        const currentScrollTop = container.scrollTop;
        setTimeout(() => {
          container.scrollTo({
            top: currentScrollTop,
            behavior: "auto",
          });
        }, 0);
      }
    }
  };

  const renderMessages = () => {
    const messageGroups = groupMessages(selectedChatMessages);

    return messageGroups.map((group, groupIndex) => (
      <div key={`group-${groupIndex}`}>
        {group.showDate && (
          <div className="text-center text-gray-500 items-center flex justify-between my-8">
            <hr
              className={`bg-slate-500 h-px border-none mx-2 ${
                showInfo ? "w-[10vw]" : "w-[20vw]"
              }`}
            />
            {moment(group.messages[0].timestamp).format("LL")}
            <hr
              className={`bg-slate-500 h-px border-none mx-2 ${
                showInfo ? "w-[10vw]" : "w-[20vw]"
              }`}
            />
          </div>
        )}

        {group.messages.map((message, messageIndex) => {
          const isLastInGroup = messageIndex === group.messages.length - 1;
          const isFirstInGroup = messageIndex === 0;

          return (
            <div key={`message-${messageIndex}`}>
              {selectedChatType === "contact" &&
                renderDM(message, isLastInGroup, isFirstInGroup)}
              {selectedChatType === "channel" &&
                renderChannelMessages(message, isLastInGroup, isFirstInGroup)}
            </div>
          );
        })}
      </div>
    ));
  };

  const downloadFile = async (url) => {
    // const response = await apiClient.get(`${HOST}/${url}`, {
    //   responseType: "blob",
    // });
    // const urlBlob = window.URL.createObjectURL(new Blob([response.data]));
    // const link = document.createElement("a");
    // link.href = urlBlob;
    // link.setAttribute("download", url.split("/").pop());
    // document.body.appendChild(link);
    // link.click();
    // link.remove();
    // window.URL.revokeObjectURL(urlBlob);
  };

  const replyMessageClick = (message) => {
    dispatch(setReplyingMessage(message));
    console.log(message);
    onReplyMessageClick();
  };

  const renderDM = (message, isLastInGroup, isFirstInGroup) => {
    const isSentByCurrentUser = message.sender === currentUser._id;

    if (isSentByCurrentUser) {
      // Current user's messages (right-aligned)
      return (
        <>
          {message.replyTo && (
            <div className="flex flex-col mt-2">
              <div className="text-xs font-normal text-slate-500 mb-1 text-right">
                Reply to {message.replyTo?.sender?.name}
              </div>
              <div className="bg-[#e0e0e0] font-normal text-slate-500 break-words rounded px-4 py-1 text-right max-w-[50%] ml-auto w-fit">
                {message.replyTo?.content}
              </div>
            </div>
          )}
          <div
            className={`${
              isLastInGroup ? "mt-0 mb-0" : "mt-0 mb-1"
            } text-right`}
          >
            {message.messageType === "text" && (
              <div className="flex items-center justify-end gap-1 group">
                {/*Different for files and image*/}
                {!message.isPinned ? (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <div
                          className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                          onClick={() => pinMessage(message)}
                        >
                          <RiPushpin2Fill />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="bg-white">
                        <p>Pin</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ) : (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <div
                          className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                          onClick={() => unpinMessage(message)}
                        >
                          <RiUnpinFill />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent className="bg-white">
                        <p>Unpin</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger className="group-hover:block hidden">
                      <ForwardMessage currentMessageContent={message.content} />
                    </TooltipTrigger>
                    <TooltipContent className="bg-white">
                      <p>Forward</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <div
                        className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                        onClick={() => replyMessageClick(message)}
                      >
                        <RiReplyFill />
                      </div>
                    </TooltipTrigger>
                    <TooltipContent className="bg-white">
                      <p>Reply</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <div className="border-[#26597C] bg-[#26597C] text-white inline-block max-w-[50%] break-words rounded border px-4 py-2">
                  {message.content}
                </div>
              </div>
            )}
            {message.messageType === "file" && (
              <div className="bg-[#4CB8FF]/5 text-black/90 inline-block max-w-[50%] break-words rounded border p-4">
                {/* File content rendering remains the same */}
                {checkIfImage(message.fileUrl) ? (
                  <div
                    className="cursor-pointer"
                    onClick={() => {
                      setShowImage(true);
                      setImageURL(message.fileUrl);
                    }}
                  >
                    {/* Image rendering code remains the same */}
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-4">
                    <span className="text-white/8 rounded-full bg-black/20 p-3 text-3xl">
                      <MdAttachFile className="cursor-pointer" />
                    </span>
                    <span className="cursor-pointer">
                      {message.fileUrl.split("/").pop()}
                    </span>
                    <span
                      className="cursor-pointer rounded-full p-3 text-xl transition-all duration-300 hover:bg-blue-400"
                      onClick={() => downloadFile(message.fileUrl)}
                    >
                      <IoArrowDownOutline />
                    </span>
                  </div>
                )}
              </div>
            )}
            {isLastInGroup && (
              <div className="mt-1 text-xs text-black/60 mb-4">
                {moment(message.timestamp).format("LT")}
              </div>
            )}
          </div>
        </>
      );
    } else {
      // Other users' messages (left-aligned with avatar)
      return (
        <>
          <div className={`${isLastInGroup ? "mt-0 mb-0" : "mt-0 mb-1"}`}>
            {/* Sender name at the top for the first message in a group */}
            {isFirstInGroup && (
              <div className="flex items-center justify-start mb-1 ml-[48px]">
                <span className="text-sm font-medium text-black/80">
                  {selectedChatData.name}
                </span>
              </div>
            )}

            {message.replyTo && (
              <div className="flex flex-col ml-[48px] mt-2">
                <div className="text-xs font-normal text-slate-500 mb-1">
                  Reply to{" "}
                  {currentUser.name === message.replyTo?.sender?.name
                    ? "you"
                    : message.replyTo?.sender?.name}
                </div>
                <div className="bg-[#e0e0e0] font-normal text-slate-500 rounded px-4 py-1 break-words max-w-[50%] inline-block w-fit">
                  {message.replyTo?.content}
                </div>
              </div>
            )}

            {/* Message with avatar for the last message in a group */}
            <div className="flex items-start">
              {isLastInGroup ? (
                <div className="mr-2">
                  <div className="relative">
                    <Avatar className="h-[40px] w-[40px] overflow-hidden rounded-full">
                      <AvatarImage
                        src={selectedChatData.profilePicture}
                        alt="profile"
                        className="h-full w-full bg-black object-cover"
                      />
                    </Avatar>
                    <div
                      className={`rounded-full w-3 h-3 ml-2 mt-2 absolute bottom-0 right-0 ${
                        onlineUsers?.includes(selectedChatData._id)
                          ? "bg-green-500"
                          : "bg-red-500"
                      }`}
                    ></div>
                  </div>
                </div>
              ) : (
                <div className="mr-[48px]"></div> // Placeholder to maintain alignment
              )}

              <div className="w-full">
                {message.messageType === "text" && (
                  <div className="flex items-center justify-start gap-1 group">
                    <div className="bg-[#E8E8E8] text-black/90 inline-block max-w-[50%] break-words rounded border px-4 py-2">
                      {message.content}
                    </div>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <div
                            className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                            onClick={() => replyMessageClick(message)}
                          >
                            <RiReplyFill />
                          </div>
                        </TooltipTrigger>
                        <TooltipContent className="bg-white">
                          <p>Reply</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="group-hover:block hidden">
                          <ForwardMessage
                            currentMessageContent={message.content}
                          />
                        </TooltipTrigger>
                        <TooltipContent className="bg-white">
                          <p>Forward</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    {/*Different for files and image*/}
                    {!message.isPinned ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <div
                              className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                              onClick={() => pinMessage(message)}
                            >
                              <RiPushpin2Fill />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Pin</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <div
                              className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                              onClick={() => unpinMessage(message)}
                            >
                              <RiUnpinFill />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Unpin</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                )}
                {message.messageType === "file" && (
                  <div className="border-black/20 bg-[#2a2b33]/5 text-black/80 inline-block max-w-[50%] break-words rounded border p-4">
                    {checkIfImage(message.fileUrl) ? (
                      <div
                        className="cursor-pointer"
                        onClick={() => {
                          setShowImage(true);
                          setImageURL(message.fileUrl);
                        }}
                      >
                        {/* Image rendering code remains the same */}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-4">
                        <span className="text-white/8 rounded-full bg-black/20 p-3 text-3xl">
                          <MdAttachFile className="cursor-pointer" />
                        </span>
                        <span className="cursor-pointer">
                          {message.fileUrl.split("/").pop()}
                        </span>
                        <span
                          className="cursor-pointer rounded-full p-3 text-xl transition-all duration-300 hover:bg-blue-400"
                          onClick={() => downloadFile(message.fileUrl)}
                        >
                          <IoArrowDownOutline />
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Timestamp at the bottom of the last message */}
                {isLastInGroup && (
                  <div className="mt-1 text-xs text-black/60 mb-4">
                    {moment(message.timestamp).format("LT")}
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      );
    }
  };

  const renderChannelMessages = (message, isLastInGroup, isFirstInGroup) => {
    const isSentByCurrentUser = message.sender._id === currentUser._id;

    if (isSentByCurrentUser) {
      // Current user's messages (right-aligned)
      return (
        <>
          {message.replyTo && (
            <div className="flex flex-col mt-2">
              <div className="text-xs font-normal text-slate-500 mb-1 text-right">
                Reply to {message.replyTo?.sender?.name}
              </div>
              <div className="bg-[#e0e0e0] font-normal text-slate-500 break-words rounded px-4 py-1 text-right max-w-[50%] ml-auto w-fit">
                {message.replyTo?.content}
              </div>
            </div>
          )}
          <div
            className={`${
              isLastInGroup ? "mt-0 mb-0" : "mt-0 mb-1"
            } text-right`}
          >
            {message.messageType === "event" ? (
              <div className="text-center text-gray-500 items-center justify-center my-8">
                {`${message.content}`}
              </div>
            ) : (
              <>
                {message.messageType === "text" && (
                  <div className="flex items-center justify-end gap-1 group">
                    {/*Different for files and image*/}
                    {!message.isPinned ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <div
                              className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                              onClick={() => pinMessage(message)}
                            >
                              <RiPushpin2Fill />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Pin</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <div
                              className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                              onClick={() => unpinMessage(message)}
                            >
                              <RiUnpinFill />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Unpin</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="group-hover:block hidden">
                          <ForwardMessage
                            currentMessageContent={message.content}
                          />
                        </TooltipTrigger>
                        <TooltipContent className="bg-white">
                          <p>Forward</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <div
                            className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                            onClick={() => replyMessageClick(message)}
                          >
                            <RiReplyFill />
                          </div>
                        </TooltipTrigger>
                        <TooltipContent className="bg-white">
                          <p>Reply</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                    <div className="border-[#26597C] bg-[#26597C] text-white inline-block max-w-[50%] break-words rounded border px-4 py-2">
                      {message.content}
                    </div>
                  </div>
                )}
                {message.messageType === "file" && (
                  <div className="bg-[#4CB8FF]/5 text-black/90 inline-block max-w-[50%] break-words rounded border p-4">
                    {/* File content rendering remains the same */}
                    {checkIfImage(message.fileUrl) ? (
                      <div
                        className="cursor-pointer"
                        onClick={() => {
                          setShowImage(true);
                          setImageURL(message.fileUrl);
                        }}
                      >
                        {/* Image rendering code remains the same */}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-4">
                        <span className="text-white/8 rounded-full bg-black/20 p-3 text-3xl">
                          <MdAttachFile className="cursor-pointer" />
                        </span>
                        <span className="cursor-pointer">
                          {message.fileUrl.split("/").pop()}
                        </span>
                        <span
                          className="cursor-pointer rounded-full p-3 text-xl transition-all duration-300 hover:bg-blue-400"
                          onClick={() => downloadFile(message.fileUrl)}
                        >
                          <IoArrowDownOutline />
                        </span>
                      </div>
                    )}
                  </div>
                )}
                {isLastInGroup && (
                  <div className="mt-1 text-xs text-black/60 mb-4">
                    {moment(message.timestamp).format("LT")}
                  </div>
                )}
              </>
            )}
          </div>
        </>
      );
    } else {
      // Other users' messages (left-aligned with avatar)
      return (
        <div className={`${isLastInGroup ? "mt-0 mb-0" : "mt-0 mb-1"}`}>
          {/* Sender name at the top for the first message in a group */}
          {message.messageType === "event" ? (
            <div className="text-center text-gray-500 items-center justify-center my-8">
              {`${message.content}`}
            </div>
          ) : (
            <>
              {isFirstInGroup && (
                <div className="flex items-center justify-start mb-1 ml-[50px]">
                  <span className="text-sm font-medium text-black/80">
                    {message.sender.name}
                  </span>
                </div>
              )}
              {message.replyTo && (
                <div className="flex flex-col ml-[48px] mt-2">
                  <div className="text-xs font-normal text-slate-500 mb-1">
                    Reply to{" "}
                    {currentUser.name === message.replyTo?.sender?.name
                      ? "you"
                      : message.replyTo?.sender?.name}
                  </div>
                  <div className="bg-[#e0e0e0] font-normal text-slate-500 rounded px-4 py-1 break-words max-w-[50%] inline-block w-fit">
                    {message.replyTo?.content}
                  </div>
                </div>
              )}
              {/* Message with avatar for the last message in a group */}
              <div className="flex items-start">
                {isLastInGroup ? (
                  <div className="mr-2">
                    <div className="relative">
                      <Avatar className="h-[40px] w-[40px] overflow-hidden rounded-full">
                        <AvatarImage
                          src={message.sender.profilePicture}
                          alt="profile"
                          className="h-full w-full bg-black object-cover"
                        />
                      </Avatar>
                      <div
                        className={`rounded-full w-3 h-3 ml-2 mt-2 absolute bottom-0 right-0 ${
                          onlineUsers?.includes(message.sender._id)
                            ? "bg-green-500"
                            : "bg-red-500"
                        }`}
                      ></div>
                    </div>
                  </div>
                ) : (
                  <div className="mr-[48px]"></div> // Placeholder to maintain alignment
                )}

                <div className="w-full">
                  {message.messageType === "text" && (
                    <div className="flex items-center justify-start gap-1 group">
                      <div className="bg-[#E8E8E8] text-black/90 inline-block max-w-[50%] break-words rounded border px-4 py-2">
                        {message.content}
                      </div>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <div
                              className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                              onClick={() => replyMessageClick(message)}
                            >
                              <RiReplyFill />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Reply</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger className="group-hover:block hidden">
                            <ForwardMessage
                              currentMessageContent={message.content}
                            />
                          </TooltipTrigger>
                          <TooltipContent className="bg-white">
                            <p>Forward</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      {/*Different for files and image*/}
                      {!message.isPinned ? (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <div
                                className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                                onClick={() => pinMessage(message)}
                              >
                                <RiPushpin2Fill />
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="bg-white">
                              <p>Pin</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      ) : (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <div
                                className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100 group-hover:block hidden"
                                onClick={() => unpinMessage(message)}
                              >
                                <RiUnpinFill />
                              </div>
                            </TooltipTrigger>
                            <TooltipContent className="bg-white">
                              <p>Unpin</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}
                    </div>
                  )}
                  {message.messageType === "file" && (
                    <div className="border-black/20 bg-[#2a2b33]/5 text-black/80 inline-block max-w-[50%] break-words rounded border p-4">
                      {checkIfImage(message.fileUrl) ? (
                        <div
                          className="cursor-pointer"
                          onClick={() => {
                            setShowImage(true);
                            setImageURL(message.fileUrl);
                          }}
                        >
                          {/* Image rendering code remains the same */}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center gap-4">
                          <span className="text-white/8 rounded-full bg-black/20 p-3 text-3xl">
                            <MdAttachFile className="cursor-pointer" />
                          </span>
                          <span className="cursor-pointer">
                            {message.fileUrl.split("/").pop()}
                          </span>
                          <span
                            className="cursor-pointer rounded-full p-3 text-xl transition-all duration-300 hover:bg-blue-400"
                            onClick={() => downloadFile(message.fileUrl)}
                          >
                            <IoArrowDownOutline />
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Timestamp at the bottom of the last message */}
                  {isLastInGroup && (
                    <div className="mt-1 text-xs text-black/60 mb-4">
                      {moment(message.timestamp).format("LT")}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      );
    }
  };

  return (
    <div
      className="w-full flex-1 overflow-y-auto overflow-x-hidden p-4 px-8"
      //   onScroll={(e) => e.stopPropagation()} // Prevent parent scroll
    >
      {renderMessages()}
      <div ref={scrollRef} style={{ height: 1 }} />
      {showImage && (
        <div className="fixed left-0 top-0 z-[1000] flex h-[100vh] w-[100vw] flex-col items-center justify-center backdrop-blur-lg">
          {/* <div>
            <img
              src={`${HOST}/${imageURL}`}
              className="h-[80vh] w-full bg-cover"
            />
          </div> */}
          <div className="fixed top-0 mt-5 flex gap-5">
            <button
              className="cursor-pointer rounded-full p-3 text-xl transition-all duration-100 hover:bg-blue-400"
              onClick={() => downloadFile(imageURL)}
            >
              <IoArrowDownOutline />
            </button>
            <button
              className="cursor-pointer rounded-full p-3 text-xl transition-all duration-100 hover:bg-blue-400"
              onClick={() => {
                setShowImage(false);
                setImageURL(null);
              }}
            >
              <IoCloseSharp />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default MessageContainer;
