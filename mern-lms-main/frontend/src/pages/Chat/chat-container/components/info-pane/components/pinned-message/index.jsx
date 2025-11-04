import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useState, useEffect } from "react";
import { useSocket } from "@/context/SocketContext.jsx";
import { useSelector } from "react-redux";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button.jsx";
import { MdPushPin } from "react-icons/md";
import {
  GET_CHANNEL_PINNED_MESSAGES_ROUTE,
  GET_PINNED_MESSAGES_ROUTE,
} from "@/utils/constants.js";
import { apiClient } from "@/lib/api-client.js";
import { Avatar, AvatarImage } from "@/components/ui/avatar.jsx";
import moment from "moment";

function PinnedMessages({ currentContext, isChannel }) {
  const socket = useSocket().current;
  //   const { selectedChatData } = useSelector((state) => state.chat);	//TODO figure out why not working
  const { currentUser } = useSelector((state) => state.user);
  const [openPinnedMessagesModal, setOpenPinnedMessagesModal] = useState(false);
  const [pinnedMessage, setPinnedMessages] = useState([]);

  useEffect(() => {
    const getPinnedMessages = async () => {
      const response = await apiClient.post(
        GET_PINNED_MESSAGES_ROUTE,
        { id: currentContext._id },
        { withCredentials: true }
      );
      if (response.data.messages) {
        setPinnedMessages(response.data.messages);
      }
    };
    const getChannelPinnedMessages = async () => {
      const response = await apiClient.get(
        `${GET_CHANNEL_PINNED_MESSAGES_ROUTE}/${currentContext._id}`,
        { withCredentials: true }
      );
      if (response.data.messages) {
        setPinnedMessages(response.data.messages);
      }
    };

    if (openPinnedMessagesModal) {
      if (isChannel) {
        getChannelPinnedMessages();
      } else {
        getPinnedMessages();
      }
    }
  }, [currentContext, openPinnedMessagesModal]);

  return (
    <>
      <div
        className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer"
        onClick={() => setOpenPinnedMessagesModal(true)}
      >
        <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center">
          <MdPushPin className="text-black text-xl" />
        </div>
        <span className="text-lg font-medium">Pinned Messages</span>
      </div>
      <Dialog
        open={openPinnedMessagesModal}
        onOpenChange={setOpenPinnedMessagesModal}
      >
        <DialogContent className="flex w-[100vw] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Pinned Messages
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-2 p-4">
            {pinnedMessage.length > 0 ? (
              pinnedMessage.map((message) => {
                return (
                  <div
                    key={message._id}
                    className="py-2 px-2 flex rounded-md gap-5 items-center justify-start relative mr-5"
                  >
                    <div className="rounded-full flex items-center justify-center">
                      <Avatar className="h-11 w-11">
                        <AvatarImage src={message.sender.profilePicture} />
                      </Avatar>
                    </div>
                    <div className="flex flex-col gap-1 w-full">
                      <span className="text-black/60 text-sm font-semibold break-words">
                        {message.sender.name}
                      </span>
                      <span className="text-black/90 text-lg break-words max-w-[250px]">
                        {message.content}
                      </span>
                    </div>
                    <div className="text-sm text-black/60 absolute right-0">
                      {moment(message.timestamp).format("LT")}
                    </div>
                  </div>
                );
              })
            ) : (
              <span className="text-lg text-center">
                No pinned messages yet.
              </span>
            )}
          </div>
          <Button
            variant="outline"
            onClick={() => setOpenPinnedMessagesModal(false)}
            className="w-full mt-4 hover:bg-[#E4E4E4] transition-all duration-100 text-lg"
          >
            Close
          </Button>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default PinnedMessages;
