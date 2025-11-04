import { Avatar, AvatarImage } from "@/components/ui/avatar.jsx";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { useSelector } from "react-redux";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { MdPushPin } from "react-icons/md";
import { AiFillWarning } from "react-icons/ai";
import { MdPermMedia } from "react-icons/md";
import { HiUserAdd, HiUserRemove } from "react-icons/hi";
import { FaPen } from "react-icons/fa";
import { Button } from "@/components/ui/button.jsx";
import { useSocket } from "@/context/SocketContext.jsx";
import ChangeGroupName from "./components/change-group-name/index.jsx";
import MemberList from "./components/member-list/index.jsx";
import AddMember from "./components/add-member/index.jsx";
import PinnedMessages from "./components/pinned-message/index.jsx";

function InfoPane() {
  const socket = useSocket().current;
  const { selectedChatData, selectedChatType } = useSelector(
    (state) => state.chat
  );
  const { currentUser } = useSelector((state) => state.user);

  return (
    <div className="flex w-[20vw] h-full bg-white border-2 border-black rounded-md overflow-y-scroll scrollbar-none">
      <div className="p-4 w-full">
        <div className="mb-4"></div>
        {/* Chat information content goes here */}
        <div className="flex flex-col gap-2">
          <div className="flex flex-col mb-10">
            {selectedChatType === "channel" ? (
              <div className="flex mb-5 h-20 w-20 items-center justify-center rounded-full bg-[#ffffff22] border border-black mx-auto text-2xl font-semibold">
                #
              </div>
            ) : (
              <Avatar className="w-20 h-20 mb-5 border border-black mx-auto">
                <AvatarImage src={selectedChatData?.profilePicture} />
              </Avatar>
            )}
            <span className="mx-auto text-xl font-medium text-gray-800 text-center">
              {selectedChatData.name
                ? selectedChatData.name
                : selectedChatData.email}
            </span>
          </div>

          {selectedChatType === "channel" && (
            <Accordion type="single" collapsible>
              <AccordionItem className="py-2 px-2 rounded-md" value="item-1">
                <AccordionTrigger className="text-lg font-medium">
                  Group information
                </AccordionTrigger>
                <AccordionContent>
                  {selectedChatData?.admin?.includes(currentUser?._id) && (
                    <ChangeGroupName currentGroup={selectedChatData} />
                  )}
                  <PinnedMessages
                    currentContext={selectedChatData}
                    isChannel={true}
                  />

                  {/* <div className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer">
                    <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center">
                      <MdPermMedia className="text-black text-xl" />
                    </div>
                    <span className="text-lg font-medium">Shared media</span>
                  </div> */}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}
          {selectedChatType === "channel" && (
            <Accordion type="single" collapsible>
              <AccordionItem className="py-2 px-2 rounded-md" value="item-1">
                <AccordionTrigger className="text-lg font-medium">
                  People in group
                </AccordionTrigger>
                <AccordionContent>
                  <MemberList channel={selectedChatData} />
                  <AddMember currentGroup={selectedChatData} />
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}

          {selectedChatType === "contact" && (
            <Accordion type="single" collapsible>
              <AccordionItem className="py-2 px-2 rounded-md" value="item-1">
                <AccordionTrigger className="text-lg font-medium">
                  Chat information
                </AccordionTrigger>
                <AccordionContent>
                  <PinnedMessages
                    currentContext={selectedChatData}
                    isChannel={false}
                  />

                  {/* <div className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer">
                    <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center">
                      <MdPermMedia className="text-black text-xl" />
                    </div>
                    <span className="text-lg font-medium">Shared media</span>
                  </div> */}
                  {/* <div className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer">
                    <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center">
                      <AiFillWarning className="text-black text-xl" />
                    </div>
                    <span className="text-lg font-medium">Report user</span>
                  </div> */}
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}
        </div>
      </div>
    </div>
  );
}

export default InfoPane;
