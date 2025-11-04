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
import {
  RiReplyFill,
  RiShareForwardFill,
  RiShareForward2Fill,
  RiPushpin2Fill,
  RiPushpinFill,
  RiUnpinFill,
} from "react-icons/ri";
import { Checkbox } from "@/components/ui/checkbox.jsx";
import { apiClient } from "@/lib/api-client";
import { SEARCH_CONTACTS_ROUTE, SEARCH_CHANNEL_ROUTE } from "@/utils/constants";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

function ForwardMessage({ currentMessageContent }) {
  const socket = useSocket().current;
  const { currentUser } = useSelector((state) => state.user);
  const { selectedChatType } = useSelector((state) => state.chat);
  const [openForwardModal, setOpenForwardModal] = useState(false);
  const [searchedContacts, setSearchedContacts] = useState([]);
  const [selectedContacts, setSelectedContacts] = useState([]);
  const [searchedChannels, setSearchedChannels] = useState([]);
  const [selectedChannels, setSelectedChannels] = useState([]);

  const searchContacts = async (searchTerm) => {
    const response = await apiClient.post(
      `${SEARCH_CONTACTS_ROUTE}`,
      { searchTerm },
      {
        withCredentials: true,
      }
    );
    if (response.data.contacts) {
      setSearchedContacts(response.data.contacts);
    }
  };

  const searchChannels = async (searchTerm) => {
    const response = await apiClient.post(
      `${SEARCH_CHANNEL_ROUTE}`,
      { searchTerm },
      {
        withCredentials: true,
      }
    );
    if (response.data.channels) {
      setSearchedChannels(response.data.channels);
    }
  };

  const closeDialog = () => {
    setOpenForwardModal(false);
    setSearchedContacts([]);
    setSearchedChannels([]);
    setSelectedContacts([]);
    setSelectedChannels([]);
  };

  const handleForwardMessage = () => {
    if (selectedContacts.length !== 0) {
      selectedContacts.forEach((contact) => {
        socket.emit("send-message", {
          sender: currentUser._id,
          content: currentMessageContent,
          recipient: contact._id,
          messageType: "text",
          fileUrl: undefined,
          replyTo: undefined,
        });
      });
    }
    if (selectedChannels.length !== 0) {
      selectedChannels.forEach((channel) => {
        socket.emit("send-channel-message", {
          sender: currentUser._id,
          content: currentMessageContent,
          messageType: "text",
          fileUrl: undefined,
          replyTo: undefined,
          channelId: channel._id,
        });
      });
    }
    setSelectedContacts([]);
    setSelectedChannels([]);
  };

  useEffect(() => {
    if (openForwardModal) {
      searchContacts("");
      searchChannels("");
    }
  }, [openForwardModal]);

  return (
    <>
      <div
        className="hover:bg-[#E8E8E8] rounded-full p-2 transition-all duration-100"
        onClick={() => setOpenForwardModal(true)}
      >
        <RiShareForward2Fill />
      </div>
      <Dialog open={openForwardModal} onOpenChange={setOpenForwardModal}>
        <DialogContent className="flex h-[700px] w-[500px] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Forward message to
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-3 h-full">
            <div>
              <Input
                placeholder="Search contacts"
                className="rounded-lg border-2 border-black-20 bg-[#ebebeb] p-6 transition-all duration-100 focus:bg-white"
                onChange={(e) => searchContacts(e.target.value)}
              />
            </div>
            <Tabs
              defaultValue={
                selectedChatType === "contact"
                  ? "contact"
                  : selectedChatType === "channel"
                  ? "group"
                  : "contact"
              }
              className="w-full max-h-[90%]"
            >
              <TabsList className="w-full rounded-none bg-transparent p-0">
                <TabsTrigger
                  value="contact"
                  className="w-full h-full rounded-none p-3 overflow-hidden text-black text-lg transition-all duration-100 data-[state=active]:bg-[#F8F8D5] data-[state=active]:font-semibold data-[state=active]:text-black border-b-black/40 border-r-black/40 border-r border-b"
                >
                  Private
                </TabsTrigger>
                <TabsTrigger
                  value="group"
                  className="w-full h-full rounded-none p-3 overflow-hidden text-black text-lg transition-all duration-100 data-[state=active]:bg-[#F8F8D5] data-[state=active]:font-semibold data-[state=active]:text-black border-b-black/40 border-b"
                >
                  Group
                </TabsTrigger>
              </TabsList>
              <TabsContent value="contact">
                {searchedContacts.length > 0 && (
                  <ScrollArea className="h-[450px]">
                    <div className="flex flex-col gap-5">
                      {searchedContacts.map((member) => (
                        <div
                          key={member._id}
                          className="flex cursor-pointer items-center gap-3"
                          onClick={() =>
                            setSelectedContacts((selectedContacts) => {
                              const isSelected = selectedContacts.some(
                                (selectedContact) =>
                                  selectedContact._id === member._id
                              );
                              if (isSelected) {
                                return selectedContacts.filter(
                                  (selectedContact) =>
                                    selectedContact._id !== member._id
                                );
                              } else {
                                return [...selectedContacts, member];
                              }
                            })
                          }
                        >
                          <Checkbox
                            className="data-[state=checked]:bg-[#26597C] data-[state=unchecked]:hover:bg-[#dcdcdc] hover:bg-[#1d4762] text-white transition-all duration-100"
                            checked={selectedContacts.some(
                              (selectedContact) =>
                                selectedContact._id === member._id
                            )}
                          />
                          <div className="relative h-12 w-12">
                            <Avatar className="h-12 w-12 overflow-hidden rounded-full">
                              <AvatarImage
                                src={member.profilePicture}
                                alt="profile"
                                className="h-full w-full bg-black object-cover"
                              />
                            </Avatar>
                          </div>
                          <div className="flex flex-col">
                            <span>
                              {member.name ? member.name : member.email}
                            </span>
                            <span className="text-xs">{member.email}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>
              <TabsContent value="group">
                <div>
                  {searchedChannels.length > 0 && (
                    <ScrollArea className="h-[450px]">
                      <div className="flex flex-col gap-5">
                        {searchedChannels.map((channel) => (
                          <div
                            key={channel._id}
                            className="flex cursor-pointer items-center gap-3"
                            onClick={() =>
                              setSelectedChannels((selectedChannels) => {
                                const isSelected = selectedChannels.some(
                                  (selectedChannel) =>
                                    selectedChannel._id === channel._id
                                );
                                if (isSelected) {
                                  return selectedChannels.filter(
                                    (selectedChannel) =>
                                      selectedChannel._id !== channel._id
                                  );
                                } else {
                                  return [...selectedChannels, channel];
                                }
                              })
                            }
                          >
                            <Checkbox
                              className="data-[state=checked]:bg-[#26597C] data-[state=unchecked]:hover:bg-[#dcdcdc] hover:bg-[#1d4762] text-white transition-all duration-100"
                              checked={selectedChannels.some(
                                (selectedChannel) =>
                                  selectedChannel._id === channel._id
                              )}
                            />
                            <div className="h-12 w-12 bg-[#ffffff22]">
                              <div className="flex h-12 w-12 items-center justify-center rounded-full border border-black/20 flex-shrink-0">
                                #
                              </div>
                            </div>
                            <div className="flex flex-col">
                              <span>{channel.name}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            <div className="flex justify-end mt-2 gap-2 absolute right-5 bottom-4">
              <Button
                className="hover:bg-[#E4E4E4] transition-all duration-100 text-lg border"
                onClick={() => closeDialog()}
              >
                Cancel
              </Button>
              <Button
                className="text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-100 text-lg"
                onClick={() => {
                  handleForwardMessage();
                  closeDialog();
                }}
              >
                Done
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default ForwardMessage;
